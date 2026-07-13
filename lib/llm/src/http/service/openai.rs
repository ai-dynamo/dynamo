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
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::{
    RouteDoc,
    disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects},
    error::HttpError,
    metadata::{attach_x_request_id, extract_metadata_from_http},
    metrics::{
        CancellationLabels, Endpoint, ErrorType, EventConverter,
        process_chat_response_and_observe_metrics,
        process_chat_response_using_event_converter_and_observe_metrics,
        process_response_and_observe_metrics,
        process_response_using_event_converter_and_observe_metrics,
    },
    service_v2,
};
use crate::engines::ValidateRequest;
use crate::preprocessor::PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY;
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId,
    agent_context_from_headers, apply_header_routing_overrides, session_affinity_from_headers,
};
use crate::protocols::openai::chat_completions::aggregator::ChatCompletionAggregator;
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
use crate::request_template::{RequestTemplate, resolve_request_model};
use crate::types::Annotated;
use dynamo_protocols::types::ChatCompletionMessageContent;
use dynamo_protocols::types::ChatCompletionMessageToolCallChunk;
use dynamo_protocols::types::ChatCompletionStreamResponseDelta;
use dynamo_protocols::types::Choice;
use dynamo_runtime::logging::get_distributed_tracing_context;
use tracing::Instrument;

pub const DYNAMO_REQUEST_ID_HEADER: &str = "x-dynamo-request-id";

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

const VALIDATION_PREFIX: &str = "Validation: ";

use super::error::{SanitizedError, overload_status_code};

pub(super) fn rl_router(
    drt: Arc<dynamo_runtime::DistributedRuntime>,
) -> anyhow::Result<axum::Router> {
    let config = dynamo_rl::RlDiscoveryConfig::from_env(drt);
    let state = dynamo_rl::RlDiscoveryState::new(config);
    Ok(dynamo_rl::rl_router(state))
}

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
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<Box<serde_json::Value>>,
}

fn map_error_code_to_error_type(code: StatusCode) -> String {
    match code.canonical_reason() {
        Some(reason) => reason.to_string(),
        None if code.as_u16() == 529 => "Overloaded".to_string(),
        // 499 is not IANA-registered (nginx convention for client-closed-request),
        // so canonical_reason() returns None. Use the de facto standard name.
        None if code.as_u16() == 499 => "Client Closed Request".to_string(),
        None => "UnknownError".to_string(),
    }
}

/// Classify error for metrics based on status code and message
pub(super) fn classify_error_for_metrics(code: StatusCode, message: &str) -> ErrorType {
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
        StatusCode::SERVICE_UNAVAILABLE => ErrorType::Unavailable, // 503
        StatusCode::INTERNAL_SERVER_ERROR => ErrorType::Internal, // 500
        _ if code.as_u16() == 529 => ErrorType::Overload, // 529
        _ if code.as_u16() == 499 => ErrorType::Cancelled, // 499 Client Closed Request
        _ if code.is_client_error() => ErrorType::Validation, // other 4xx
        _ => ErrorType::Internal,                     // everything else
    }
}

/// Extract ErrorType from ErrorResponse for metrics
pub(super) fn extract_error_type_from_response(response: &ErrorResponse) -> ErrorType {
    classify_error_for_metrics(response.0, &response.1.message)
}

/// Match `InvalidArgument` at top-level OR under `Backend()`.
/// `py_err_to_dynamo` wraps Python `ValueError`/`TypeError` as
/// `Backend(InvalidArgument)`; both variants are 400-worthy.
fn find_invalid_argument_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
) -> Option<&'a dynamo_runtime::error::DynamoError> {
    use dynamo_runtime::error::{BackendError, ErrorType};
    let mut current = Some(err);
    while let Some(e) = current {
        if let Some(dynamo_err) = e.downcast_ref::<dynamo_runtime::error::DynamoError>()
            && matches!(
                dynamo_err.error_type(),
                ErrorType::InvalidArgument | ErrorType::Backend(BackendError::InvalidArgument)
            )
        {
            return Some(dynamo_err);
        }
        current = e.source();
    }
    None
}

fn find_queue_rejection_in_chain<'a>(
    err: &'a (dyn std::error::Error + 'static),
) -> Option<&'a dynamo_kv_router::scheduling::QueueRejection> {
    let mut current = Some(err);
    while let Some(error) = current {
        if let Some(rejection) =
            error.downcast_ref::<dynamo_kv_router::scheduling::QueueRejection>()
        {
            return Some(rejection);
        }
        current = error.source();
    }
    None
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
                details: None,
            }),
        )
    }

    /// Convert a ModelManagerError to the appropriate HTTP response.
    ///
    /// `ModelUnavailable` is the dispatch-time backstop for the same condition
    /// the readiness gate ([`check_model_serving_ready`]) catches up front — a
    /// registered model with no servable worker set (whichever role is missing).
    /// It returns the identical canonical 503 body so both code paths speak with
    /// one voice to the client.
    pub fn from_model_error(e: &crate::discovery::ModelManagerError) -> ErrorResponse {
        match e {
            crate::discovery::ModelManagerError::ModelUnavailable(model) => {
                Self::service_unavailable_with_body(model_not_ready_message(model))
            }
            _ => Self::model_not_found(),
        }
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
                details: None,
            }),
        )
    }

    /// Service Unavailable with a structured message body. Used by readiness
    /// reporting to distinguish "model registered but not ready" from generic
    /// "service not ready".
    pub fn service_unavailable_with_body(message: String) -> ErrorResponse {
        let code = StatusCode::SERVICE_UNAVAILABLE;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message,
                error_type,
                code: code.as_u16(),
                details: None,
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
                details: None,
            }),
        )
    }

    /// Internal Server Error with sanitized client message.
    /// Logs `details` server-side and returns only `public_msg` to the client.
    /// Use this whenever the detail could carry an anyhow chain, JoinError
    /// debug output, or anything else that may leak file paths, library
    /// versions, or other internal implementation details.
    pub fn internal_server_error_with_details(
        public_msg: &str,
        details: impl std::fmt::Display,
    ) -> ErrorResponse {
        tracing::error!("Internal server error: {public_msg}: {details}");
        let code = StatusCode::INTERNAL_SERVER_ERROR;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: public_msg.to_string(),
                error_type,
                code: code.as_u16(),
                details: None,
            }),
        )
    }

    /// Build a sanitized error response from a [`SanitizedError`] variant.
    /// The status, public message, and protocol error_type all come from
    /// the variant — call sites do not pass any of them as literals.
    /// Server-side `details` are logged alongside the canonical category;
    /// the client only ever sees the variant's public message.
    pub fn sanitized_with_details(
        err: SanitizedError,
        details: impl std::fmt::Display,
    ) -> ErrorResponse {
        let status = err.status();
        if err.log_as_error() {
            tracing::error!(status = %status, "{err}: {details}");
        } else {
            tracing::debug!(status = %status, "{err}: {details}");
        }
        (
            status,
            Json(ErrorMessage {
                message: err.to_string(),
                error_type: map_error_code_to_error_type(status),
                code: status.as_u16(),
                details: None,
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
                details: None,
            }),
        )
    }

    pub fn request_headers_too_large(msg: &str) -> ErrorResponse {
        let code = StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE;
        let error_type = map_error_code_to_error_type(code);
        (
            code,
            Json(ErrorMessage {
                message: msg.to_string(),
                error_type,
                code: code.as_u16(),
                details: None,
            }),
        )
    }

    /// The OAI endpoints call an [`dynamo.runtime::engine::AsyncEngine`] which are specialized to return
    /// an [`anyhow::Error`]. This method will convert the [`anyhow::Error`] into an [`HttpError`].
    /// If successful, it will return the [`HttpError`] as an [`ErrorMessage::internal_server_error`]
    /// with the details of the error.
    pub fn from_anyhow(err: anyhow::Error, alt_msg: &str) -> ErrorResponse {
        if let Some(rejection) = find_queue_rejection_in_chain(err.as_ref()) {
            let code = overload_status_code();
            return (
                code,
                Json(ErrorMessage {
                    message: rejection.to_string(),
                    error_type: map_error_code_to_error_type(code),
                    code: code.as_u16(),
                    details: serde_json::to_value(rejection).ok().map(Box::new),
                }),
            );
        }

        // Check for ResourceExhausted anywhere in the error chain → HTTP 529
        if super::metrics::request_was_rejected(err.as_ref()) {
            return ErrorMessage::sanitized_with_details(
                SanitizedError::Overloaded,
                format!("{err:#}"),
            );
        }

        // No backend workers are currently routable → HTTP 503.
        if super::metrics::request_was_unavailable(err.as_ref()) {
            return ErrorMessage::sanitized_with_details(
                SanitizedError::Unavailable,
                format!("{err:#}"),
            );
        }

        // InvalidArgument (top-level OR Backend) → 400.
        if let Some(dynamo_err) = find_invalid_argument_in_chain(err.as_ref()) {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorMessage {
                    message: dynamo_err.message().to_string(),
                    error_type: map_error_code_to_error_type(StatusCode::BAD_REQUEST),
                    code: StatusCode::BAD_REQUEST.as_u16(),
                    details: None,
                }),
            );
        }

        // Check for Cancelled anywhere in the error chain → HTTP 499 (Client Closed Request)
        if super::metrics::request_was_cancelled(err.as_ref()) {
            return ErrorMessage::sanitized_with_details(
                SanitizedError::Cancelled,
                format!("{err:#}"),
            );
        }

        // Then check for HttpError
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorMessage::from_http_error(http_error),
            Err(err) => {
                ErrorMessage::internal_server_error_with_details(alt_msg, format!("{err:#}"))
            }
        }
    }

    /// Implementers should only be able to throw 400-499 errors.
    pub fn from_http_error(err: HttpError) -> ErrorResponse {
        // 499 is part of the 4xx range but its body can carry cancellation
        // context (queue paths, context IDs) — sanitize separately.
        if err.code == 499 {
            return ErrorMessage::sanitized_with_details(SanitizedError::Cancelled, err.message);
        }
        // Backend-supplied messages are only forwarded for the documented 4xx
        // range; for 5xx or codes outside the HTTP space the message may
        // contain internal paths/details and is kept server-side only.
        if err.code < 400 || err.code >= 500 {
            return ErrorMessage::sanitized_with_details(SanitizedError::Internal, err.message);
        }
        match StatusCode::from_u16(err.code) {
            Ok(code) => (
                code,
                Json(ErrorMessage {
                    message: err.message,
                    error_type: map_error_code_to_error_type(code),
                    code: code.as_u16(),
                    details: None,
                }),
            ),
            Err(_) => ErrorMessage::sanitized_with_details(SanitizedError::Internal, err.message),
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
            details: None,
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
                details: None,
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

pub(super) fn context_from_headers<T: Send + Sync + 'static>(
    request: T,
    request_id: String,
    headers: &HeaderMap,
) -> Result<Context<T>, ErrorResponse> {
    let metadata = extract_metadata_from_http(headers)
        .map_err(|err| ErrorMessage::request_headers_too_large(&err.to_string()))?;
    let mut request = Context::with_id_and_metadata(request, request_id, metadata);
    attach_x_request_id(&mut request, headers);
    if let Some(agent_context) = agent_context_from_headers(headers) {
        request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context);
    }
    if let Some(session_affinity) = session_affinity_from_headers(headers) {
        request.insert(SESSION_AFFINITY_CONTEXT_KEY, session_affinity);
    }
    Ok(request)
}

fn copy_context_metadata<T: Send + Sync + 'static, U: Send + Sync + 'static>(
    source: &Context<T>,
    target: &mut Context<U>,
) {
    if crate::request_trace::is_enabled()
        && let Ok(x_request_id) =
            source.get::<String>(crate::request_trace::X_REQUEST_ID_CONTEXT_KEY)
    {
        target.insert(
            crate::request_trace::X_REQUEST_ID_CONTEXT_KEY,
            x_request_id.as_ref().clone(),
        );
    }

    if let Ok(agent_context) = source.get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY) {
        target.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context.as_ref().clone());
    }
    if let Ok(session_affinity) = source.get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY) {
        target.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            session_affinity.as_ref().clone(),
        );
    }
}

/// Warn (once per request) when nvext data is dropped because the extension is
/// disabled. Only called from the disabled branch, so the default path is free.
fn warn_nvext_disabled(endpoint: &str, nvext_present: bool, headers: &HeaderMap) {
    use crate::protocols::common::extensions::{
        HEADER_DATA_PARALLEL_RANK_ALIAS, HEADER_DP_RANK, HEADER_DP_RANK_ALIAS,
        HEADER_PREFILL_DP_RANK, HEADER_PREFILL_DP_RANK_ALIAS, HEADER_PREFILL_INSTANCE_ID,
        HEADER_PREFILL_INSTANCE_ID_ALIAS, HEADER_REQUEST_PRIORITY, HEADER_REQUEST_STRICT_PRIORITY,
        HEADER_WORKER_INSTANCE_ID, HEADER_WORKER_INSTANCE_ID_ALIAS,
    };
    let header_present = [
        HEADER_WORKER_INSTANCE_ID,
        HEADER_WORKER_INSTANCE_ID_ALIAS,
        HEADER_PREFILL_INSTANCE_ID,
        HEADER_PREFILL_INSTANCE_ID_ALIAS,
        HEADER_DP_RANK,
        HEADER_DP_RANK_ALIAS,
        HEADER_DATA_PARALLEL_RANK_ALIAS,
        HEADER_PREFILL_DP_RANK,
        HEADER_PREFILL_DP_RANK_ALIAS,
        HEADER_REQUEST_PRIORITY,
        HEADER_REQUEST_STRICT_PRIORITY,
    ]
    .iter()
    .any(|h| headers.contains_key(*h));

    if nvext_present || header_present {
        tracing::warn!(
            endpoint,
            "request carried nvext data but the nvext extension is disabled on this frontend; dropping it"
        );
    }
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
    body: Bytes,
) -> Result<Response, ErrorResponse> {
    ensure_json_content_type(&headers)?;
    let mut request: NvCreateCompletionRequest = parse_json_request("completions", &body)?;

    // return a 503 if the service or model is not ready
    check_ready(&state)?;
    check_model_serving_ready(&state, &request.inner.model)?;

    request.nvext = if state.nvext_enabled() {
        apply_header_routing_overrides(request.nvext.take(), &headers)
    } else {
        warn_nvext_disabled("completions", request.nvext.is_some(), &headers);
        None
    };

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    // Canonicalize alias → primary for the metric label.
    let canonical_model = state.manager().resolve_canonical_name(&request.inner.model);
    let cancellation_labels = CancellationLabels {
        model: state
            .manager()
            .metric_model_for(&canonical_model)
            .to_string(),
        endpoint: Endpoint::Completions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = context_from_headers(request, request_id, &headers)?;
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
            ErrorMessage::internal_server_error_with_details(
                "Failed to await chat completions task",
                format!("{e:?}"),
            )
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
    mut request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
) -> Result<Response, ErrorResponse> {
    let request_id = request.id().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    // Resolve an alias to its primary served name and rewrite the request so
    // engine routing, metrics, and the OpenAI response.model all use the
    // canonical primary (matching vLLM/SGLang, where an alias request still
    // responds with the primary served name). Non-aliases pass through, so
    // metric_model_for still applies its unknown-model cardinality guard.
    let canonical = state.manager().resolve_canonical_name(&request.inner.model);
    if canonical != request.inner.model {
        request.inner.model = canonical;
    }
    let model = request.inner.model.clone();
    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::Completions,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    // todo - error handling should be more robust
    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|e| {
            let err_response = ErrorMessage::from_model_error(&e);
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::Completions);
        }
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
            .filter(|r| {
                // Drop empty chunks from multi-byte token assembly
                futures::future::ready(
                    !r.data
                        .as_ref()
                        .is_some_and(is_empty_completion_stream_response),
                )
            })
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
                    "Failed to fold completions stream for {request_id}"
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
    mut request: Context<NvCreateCompletionRequest>,
    stream_handle: ConnectionHandle,
    batch_size: usize,
    n: u8,
) -> Result<Response, ErrorResponse> {
    use crate::protocols::openai::completions::extract_single_prompt;
    use futures::stream::{self, StreamExt};

    let request_id = request.id().to_string();
    let streaming = request.inner.stream.unwrap_or(false);
    // Resolve alias → primary served name (see completions_single).
    let canonical = state.manager().resolve_canonical_name(&request.inner.model);
    if canonical != request.inner.model {
        request.inner.model = canonical;
    }
    let model = request.inner.model.clone();
    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::Completions,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    let (engine, parsing_options) = state
        .manager()
        .get_completions_engine_with_parsing(&model)
        .map_err(|e| {
            let err_response = ErrorMessage::from_model_error(&e);
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);

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
        let mut single_request_context = Context::with_id_and_metadata(
            single_request,
            unique_request_id,
            request.metadata().clone(),
        );
        copy_context_metadata(&request, &mut single_request_context);

        // Generate stream for this prompt
        let stream = engine.generate(single_request_context).await.map_err(|e| {
            if super::metrics::request_was_rejected(e.as_ref()) {
                state
                    .metrics_clone()
                    .inc_rejection(&model, super::metrics::Endpoint::Completions);
            }
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
            .filter(|r| {
                // Drop empty chunks from multi-byte token assembly
                futures::future::ready(
                    !r.data
                        .as_ref()
                        .is_some_and(is_empty_completion_stream_response),
                )
            })
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
                    "Failed to fold completions stream for {request_id}"
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
    Json(mut request): Json<NvCreateEmbeddingRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service or model is not ready
    check_ready(&state)?;
    check_model_serving_ready(&state, &request.inner.model)?;

    if !state.nvext_enabled() {
        warn_nvext_disabled("embeddings", request.nvext.is_some(), &headers);
        request.nvext = None;
    }

    // Resolve alias → primary served name before wrapping the request, so
    // engine routing, metrics, and the response model all use the canonical
    // primary (see completions_single). `request` is still owned + mutable here.
    let canonical = state.manager().resolve_canonical_name(&request.inner.model);
    if canonical != request.inner.model {
        request.inner.model = canonical;
    }
    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
    let request_id = request.id().to_string();

    // The worker always emits base64-encoded vectors over NATS so we
    // avoid serializing/parsing a JSON float array on the internal hop.
    // If the client asked for float (the default), decode back at the
    // HTTP boundary so the public response shape matches their
    // ``encoding_format`` choice. See the PR description / DIS-2154 for
    // measured impact.
    // Borrow rather than move ``encoding_format`` out of ``request`` so the
    // request value remains intact for the later ``engine.generate(request)``
    // call below.
    let client_wants_float = !matches!(
        request.inner.encoding_format.as_ref(),
        Some(dynamo_protocols::types::EncodingFormat::Base64)
    );

    // Embeddings are typically not streamed, so we default to non-streaming
    let streaming = false;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;
    let metric_model = state.manager().metric_model_for(model).to_string();

    // Start the embedding-specific latency timer. Distinct from
    // `request_duration` (which has 1..512s LLM-gen buckets); pooling-model
    // requests are sub-second and need finer-grained buckets to be useful for
    // SLO tracking. Only observed on the success path -- error/cancel paths
    // already increment requests_total with status=error.
    let embedding_start = std::time::Instant::now();

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::Embeddings,
        streaming,
        &request_id,
    );

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    // todo - error handling should be more robust
    let engine = state.manager().get_embeddings_engine(model).map_err(|e| {
        let err_response = ErrorMessage::from_model_error(&e);
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);
    let model_name = model.to_string();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model_name, super::metrics::Endpoint::Embeddings);
        }
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
    let mut response = NvCreateEmbeddingResponse::from_annotated_stream(stream)
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

    // Worker always emits Base64 -- convert back to Float when the client
    // asked for float (or didn't specify, defaulting to float per spec).
    if client_wants_float {
        for embedding_obj in response.inner.data.iter_mut() {
            if let dynamo_protocols::types::EmbeddingVector::Base64(s) = &embedding_obj.embedding {
                match decode_base64_embedding_to_floats(s) {
                    Ok(floats) => {
                        embedding_obj.embedding =
                            dynamo_protocols::types::EmbeddingVector::Float(floats);
                    }
                    Err(e) => {
                        tracing::error!(
                            "Failed to decode base64 embedding for request {}: {:?}",
                            request_id,
                            e
                        );
                        let err_response = ErrorMessage::internal_server_error(
                            "Failed to decode embedding payload",
                        );
                        inflight.mark_error(extract_error_type_from_response(&err_response));
                        return Err(err_response);
                    }
                }
            }
        }
    }

    state
        .metrics_clone()
        .observe_embedding_latency(&model_name, embedding_start.elapsed().as_secs_f64());
    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// Decode a base64-encoded little-endian f32 byte string back into a float
/// vector. The byte length must be a multiple of 4; trailing bytes are
/// rejected. Mirrors the encoder in `lib/llm/src/preprocessor.rs` and the
/// Python `_encode_floats_to_base64` helper in
/// `components/src/dynamo/vllm/handlers.py`.
fn decode_base64_embedding_to_floats(s: &str) -> Result<Vec<f32>, anyhow::Error> {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    let bytes = STANDARD.decode(s)?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        anyhow::bail!(
            "base64-decoded byte length {} is not a multiple of 4",
            bytes.len()
        );
    }
    let mut floats = Vec::with_capacity(bytes.len() / std::mem::size_of::<f32>());
    for chunk in bytes.chunks_exact(std::mem::size_of::<f32>()) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(floats)
}

async fn handler_chat_completions(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ErrorResponse> {
    ensure_json_content_type(&headers)?;
    let mut request: NvCreateChatCompletionRequest = parse_json_request("chat completions", &body)?;

    // return a 503 if the service is not ready (process-level + per-model
    // serving readiness). An aggregated request to a decode-only namespace
    // would otherwise hang/crash on the decode worker. Resolve the templated
    // model first so empty/missing `model` fields don't bypass the gate.
    check_ready(&state)?;
    let resolved_model = resolve_request_model(&request.inner.model, template.as_ref());
    if !resolved_model.is_empty() {
        check_model_serving_ready(&state, resolved_model)?;
    }

    request.nvext = if state.nvext_enabled() {
        apply_header_routing_overrides(request.nvext.take(), &headers)
    } else {
        warn_nvext_disabled("chat_completions", request.nvext.is_some(), &headers);
        None
    };

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let resolved_model = resolve_request_model(&request.inner.model, template.as_ref());
    // Canonicalize alias → primary for the metric label.
    let canonical_model = state.manager().resolve_canonical_name(resolved_model);
    let cancellation_labels = CancellationLabels {
        model: state
            .manager()
            .metric_model_for(&canonical_model)
            .to_string(),
        endpoint: Endpoint::ChatCompletions.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let mut request = context_from_headers(request, request_id, &headers)?;
    if let Some(captured) = crate::request_trace::payload::capture_http_headers(&headers) {
        request.insert(
            crate::request_trace::payload::HTTP_HEADERS_CONTEXT_KEY,
            captured,
        );
    }
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
                ErrorMessage::internal_server_error_with_details(
                    "Failed to await chat completions task",
                    format!("{e:?}"),
                )
            })?;

    // if we got here, then we will return a response and the potentially long running task has completed successfully
    // without need to be cancelled.
    connection_handle.disarm();

    response
}

fn parse_json_request<T>(endpoint: &'static str, body: &[u8]) -> Result<T, ErrorResponse>
where
    T: DeserializeOwned,
{
    match serde_json::from_slice(body) {
        Ok(request) => Ok(request),
        Err(original_error) => {
            if let Some(escaped_body) = escape_json_string_control_chars(body) {
                match serde_json::from_slice(&escaped_body) {
                    Ok(request) => {
                        tracing::warn!(
                            endpoint,
                            "Accepted request after escaping unescaped control characters in JSON strings"
                        );
                        Ok(request)
                    }
                    Err(_) => parse_json_request_lossy(endpoint, body)
                        .map_err(|_| json_deserialize_error(original_error)),
                }
            } else {
                parse_json_request_lossy(endpoint, body)
                    .map_err(|_| json_deserialize_error(original_error))
            }
        }
    }
}

fn parse_json_request_lossy<T>(endpoint: &'static str, body: &[u8]) -> Result<T, serde_json::Error>
where
    T: DeserializeOwned,
{
    let lossy_body = String::from_utf8_lossy(body);
    if lossy_body.as_bytes() == body {
        return serde_json::from_slice(body);
    }

    let escaped_body = escape_json_string_control_chars(lossy_body.as_bytes())
        .unwrap_or_else(|| lossy_body.into_owned().into_bytes());
    let request = serde_json::from_slice(&escaped_body)?;
    tracing::warn!(
        endpoint,
        "Accepted request after replacing invalid UTF-8 and escaping unescaped control characters in JSON strings"
    );
    Ok(request)
}

fn json_deserialize_error(error: serde_json::Error) -> ErrorResponse {
    let code = StatusCode::BAD_REQUEST;
    (
        code,
        Json(ErrorMessage {
            message: format!("Failed to deserialize the JSON body into the target type: {error}"),
            error_type: map_error_code_to_error_type(code),
            code: code.as_u16(),
            details: None,
        }),
    )
}

fn ensure_json_content_type(headers: &HeaderMap) -> Result<(), ErrorResponse> {
    let Some(content_type) = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return Err(unsupported_media_type_error());
    };

    if is_json_content_type(content_type) {
        Ok(())
    } else {
        Err(unsupported_media_type_error())
    }
}

fn unsupported_media_type_error() -> ErrorResponse {
    let code = StatusCode::UNSUPPORTED_MEDIA_TYPE;
    (
        code,
        Json(ErrorMessage {
            message: "Expected request with Content-Type application/json".to_string(),
            error_type: map_error_code_to_error_type(code),
            code: code.as_u16(),
            details: None,
        }),
    )
}

fn is_json_content_type(content_type: &str) -> bool {
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    let Some((media_type, subtype)) = media_type.split_once('/') else {
        return false;
    };

    media_type.eq_ignore_ascii_case("application")
        && (subtype.eq_ignore_ascii_case("json")
            || subtype
                .to_ascii_lowercase()
                .rsplit_once('+')
                .is_some_and(|(_, suffix)| suffix == "json"))
}

fn escape_json_string_control_chars(body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(body.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;

    for &byte in body {
        if in_string && byte <= 0x1f {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            if escaped {
                out.extend_from_slice(b"\\\\u00");
                escaped = false;
            } else {
                out.extend_from_slice(b"\\u00");
            }
            out.push(HEX[(byte >> 4) as usize]);
            out.push(HEX[(byte & 0x0f) as usize]);
            changed = true;
            continue;
        }

        out.push(byte);

        if escaped {
            escaped = false;
        } else if in_string && byte == b'\\' {
            escaped = true;
        } else if byte == b'"' {
            in_string = !in_string;
        }
    }

    changed.then_some(out)
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
        use dynamo_runtime::error::{BackendError, ErrorType};

        // Classify only this event's error, not its causes. An inner invalid
        // argument must not override an outer unavailable or internal error.
        let invalid_argument = event.error.as_ref().filter(|error| {
            matches!(
                error.error_type(),
                ErrorType::InvalidArgument | ErrorType::Backend(BackendError::InvalidArgument)
            )
        });

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

        // Parse the status-bearing node's own message. The diagnostic string
        // above includes its causes and therefore is not necessarily JSON.
        let status_message = event
            .error
            .as_ref()
            .map(|error| error.message())
            .unwrap_or(&error_str);
        if let Ok(error_payload) = serde_json::from_str::<ErrorPayload>(status_message) {
            // Preserve explicit HTTP-like statuses (for example 415); Python
            // 4xx exceptions share the Backend(InvalidArgument) category.
            let code = match error_payload.code {
                Some(code) => {
                    StatusCode::from_u16(code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                }
                None if invalid_argument.is_some() => StatusCode::BAD_REQUEST,
                None => StatusCode::INTERNAL_SERVER_ERROR,
            };
            let message = error_payload
                .message
                .unwrap_or_else(|| status_message.to_string());
            return Some((message, code));
        }

        if let Some(invalid_argument) = invalid_argument {
            return Some((
                invalid_argument.message().to_string(),
                StatusCode::BAD_REQUEST,
            ));
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

/// Returns true for events that only carry an annotation tag (e.g. the
/// `request_id` frame prepended to every stream): no data, no error, and
/// an `event` field that is *not* the `"error"` marker. Annotations may
/// still carry a serialized value in `comment` (that is how
/// `Annotated::from_annotation` builds them), so the comment field is
/// not part of the check. These frames are stepped over by
/// `check_for_backend_error` so an immediate backend error in the *next*
/// slot is still caught instead of slipping through to the fold/parse
/// path.
fn is_annotation_frame<T>(e: &Annotated<T>) -> bool {
    e.data.is_none()
        && e.error.is_none()
        && matches!(e.event.as_deref(), Some(tag) if tag != "error")
}

/// Cap on how many leading annotation frames `check_for_backend_error`
/// will buffer before giving up the inspection. A pathological backend
/// (or attacker who can influence the engine output) that emits only
/// annotation frames must not be able to pin unbounded memory per
/// request. The handful of real annotations a frontend prepends
/// (currently just `request_id`) fits well under this cap.
const MAX_LEADING_ANNOTATIONS: usize = 16;

/// Inspect the first non-annotation event in the stream for a backend error.
/// Returns Err(ErrorResponse) if error detected, Ok(stream) otherwise — the
/// returned stream replays any buffered annotation frames in their original
/// order before yielding the remaining items.
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

    let mut buffered: Vec<Annotated<NvCreateChatCompletionStreamResponse>> = Vec::new();
    while let Some(event) = stream.next().await {
        if is_annotation_frame(&event) && buffered.len() < MAX_LEADING_ANNOTATIONS {
            buffered.push(event);
            continue;
        }
        if let Some((error_msg, status_code)) = extract_backend_error_if_present(&event) {
            return Err(match SanitizedError::for_backend_status(status_code) {
                Some(variant) => ErrorMessage::sanitized_with_details(variant, error_msg),
                // 4xx (non-499): protocol contract — forward backend message as-is.
                None => (
                    status_code,
                    Json(ErrorMessage {
                        message: error_msg,
                        error_type: map_error_code_to_error_type(status_code),
                        code: status_code.as_u16(),
                        details: None,
                    }),
                ),
            });
        }

        // First non-annotation, non-error event — push it back and stop;
        // downstream consumers see the original ordering.
        buffered.push(event);
        break;
    }
    Ok(futures::stream::iter(buffered).chain(stream))
}

#[derive(Serialize)]
struct ToolCallDispatchPayload<'a> {
    choice_index: u32,
    tool_call: &'a ChatCompletionMessageToolCallChunk,
}

#[derive(Serialize)]
struct ReasoningDispatchPayload<'a> {
    index: u32,
    reasoning_content: &'a str,
}

/// Serialize `payload` and append it as an SSE event with the given name.
fn push_dispatch_event(
    event_name: &str,
    payload: &impl serde::Serialize,
    out: &mut Vec<Result<Event, axum::Error>>,
) {
    match serde_json::to_string(payload) {
        Ok(json) => out.push(Ok(Event::default().event(event_name).data(json))),
        Err(e) => {
            tracing::warn!("streaming_{event_name}: failed to serialize: {e}");
        }
    }
}

/// Empty stream chunk produced by multi-byte token assembly (e.g. emoji).
/// `role` is excluded; backends set it on every delta.
fn is_empty_stream_response(resp: &NvCreateChatCompletionStreamResponse) -> bool {
    if resp.nvext.is_some() {
        return false;
    }
    resp.inner.usage.is_none()
        && resp.inner.choices.iter().all(|c| {
            let ChatCompletionStreamResponseDelta {
                content,
                function_call,
                tool_calls,
                role: _,
                refusal,
                reasoning_content,
            } = &c.delta;
            // `Text("")` happens during multi-byte UTF-8 token assembly;
            // `Parts(vec![])` is a structurally empty multimodal payload.
            let content_empty = match content {
                None => true,
                Some(ChatCompletionMessageContent::Text(t)) => t.is_empty(),
                Some(ChatCompletionMessageContent::Parts(p)) => p.is_empty(),
            };
            c.finish_reason.is_none()
                && c.logprobs.is_none()
                && content_empty
                && function_call.is_none()
                && tool_calls.is_none()
                && refusal.is_none()
                && reasoning_content.is_none()
        })
}

/// Completions variant of [`is_empty_stream_response`].
fn is_empty_completion_stream_response(resp: &NvCreateCompletionResponse) -> bool {
    if resp.nvext.is_some() {
        return false;
    }
    resp.inner.usage.is_none()
        && resp.inner.choices.iter().all(|c| {
            let Choice {
                text,
                index: _,
                logprobs,
                finish_reason,
            } = c;
            text.is_empty() && finish_reason.is_none() && logprobs.is_none()
        })
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
    out: &mut Vec<Result<Event, axum::Error>>,
) {
    let Some(data) = &response.data else {
        return;
    };

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
                let payload = ToolCallDispatchPayload {
                    choice_index: choice.index,
                    tool_call: chunk,
                };
                push_dispatch_event("tool_call_dispatch", &payload, out);
            }
        }
    }
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
    out: &mut Vec<Result<Event, axum::Error>>,
) {
    let Some(data) = &response.data else {
        return;
    };

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
            let payload = ReasoningDispatchPayload {
                index: choice.index,
                reasoning_content: buffer.as_str(),
            };
            push_dispatch_event("reasoning_dispatch", &payload, out);
            buffer.clear();
        }
    }
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
    // Resolve an alias to its primary served name and rewrite the request so
    // engine routing, metrics, and the OpenAI response.model all use the
    // canonical primary (matching vLLM/SGLang). Non-aliases pass through so
    // metric_model_for still applies its unknown-model cardinality guard.
    let canonical = state.manager().resolve_canonical_name(&request.inner.model);
    if canonical != request.inner.model {
        request.inner.model = canonical;
    }
    let model = request.inner.model.clone();
    let metric_model = state.manager().metric_model_for(&model).to_string();

    tracing::trace!("Received chat completions request: {:?}", request.content());

    // Create inflight_guard early to ensure all errors (including validation) are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::ChatCompletions,
        streaming,
        &request_id,
    );

    if let Err(err_response) = normalize_chat_reasoning_template_args(&mut request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

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
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    // Let backend adapters apply their own generation default (e.g. --override-generation-config).
    if request.inner.max_completion_tokens.is_none() {
        request.insert(PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY, true);
    }

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|e| {
            let err_response = ErrorMessage::from_model_error(&e);
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    // Gate the experimental v2 batch finalize on the request's tool_choice, mirroring the
    // streaming gate (required/named + structural-tag stay on the v1 finalize path).
    let parsing_options = parsing_options.with_experimental_v2_batch_eligible(
        crate::protocols::openai::chat_completions::tool_parser_v2::batch_tool_choice_eligible(
            request.inner.tool_choice.as_ref(),
        ),
    );

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);

    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::ChatCompletions);
        }
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

        // Optionally prepend extra SSE events before each regular chunk:
        //   - `event: tool_call_dispatch`  — complete tool call detected early (tool dispatch)
        //   - `event: reasoning_dispatch`  — complete reasoning block (emitted once)
        let stream = async_stream::stream! {
            let mut stream = Box::pin(stream);
            let mut events: Vec<Result<Event, axum::Error>> = Vec::with_capacity(4);

            while let Some(response) = stream.next().await {
                events.clear();

                // Drop empty chunks from multi-byte token assembly.
                if response.data.as_ref().is_some_and(is_empty_stream_response) {
                    continue;
                }
                if tool_dispatch_enabled {
                    streaming_tool_dispatch_events(
                        &response,
                        &mut dispatched_tool_ids,
                        &mut events,
                    );
                }
                if reasoning_dispatch_enabled {
                    accumulate_reasoning_dispatch(
                        &response,
                        &mut reasoning_buffer,
                        &mut events,
                    );
                }

                // Convert to SSE event (this consumes the response).
                // EventConverter will detect `event: "error"` and convert to SSE error events.
                let sse_result = process_chat_response_using_event_converter_and_observe_metrics(
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

                events.reverse();
                while let Some(event) = events.pop() {
                    yield event;
                }
            }
        };
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
            process_chat_response_and_observe_metrics(
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
                    let err_response = ErrorMessage::internal_server_error(
                        "Failed to parse chat completion response",
                    );
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

/// Normalizes OpenAI-style reasoning controls before chat completion validation.
fn normalize_chat_reasoning_template_args(
    request: &mut NvCreateChatCompletionRequest,
) -> Result<(), ErrorResponse> {
    request.normalize_reasoning_template_args().map_err(|e| {
        ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: VALIDATION_PREFIX.to_string() + &e.to_string(),
        })
    })
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
    // return a 503 if the service or model is not ready.
    // Resolve the templated model first so empty/missing `model` fields
    // don't bypass the gate.
    check_ready(&state)?;
    let resolved_model = resolve_request_model(
        request.inner.model.as_deref().unwrap_or(""),
        template.as_ref(),
    );
    if !resolved_model.is_empty() {
        check_model_serving_ready(&state, resolved_model)?;
    }

    request.nvext = if state.nvext_enabled() {
        apply_header_routing_overrides(request.nvext.take(), &headers)
    } else {
        warn_nvext_disabled("responses", request.nvext.is_some(), &headers);
        None
    };

    // create the context for the request
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.inner.stream.unwrap_or(false);
    let raw_model = request.inner.model.as_deref().unwrap_or("");
    let resolved_model = resolve_request_model(raw_model, template.as_ref());
    // Canonicalize alias → primary for the metric label.
    let canonical_model = state.manager().resolve_canonical_name(resolved_model);
    let cancellation_labels = CancellationLabels {
        model: state
            .manager()
            .metric_model_for(&canonical_model)
            .to_string(),
        endpoint: Endpoint::Responses.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let mut request = context_from_headers(request, request_id, &headers)?;
    if let Some(captured) = crate::request_trace::payload::capture_http_headers(&headers) {
        request.insert(
            crate::request_trace::payload::HTTP_HEADERS_CONTEXT_KEY,
            captured,
        );
    }
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
                ErrorMessage::internal_server_error_with_details(
                    "Failed to await responses task",
                    format!("{e:?}"),
                )
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

    // Apply template values if present. When no template and no client-supplied
    // max_output_tokens, leave it as None for response echoing and let the
    // backend adapter compute the dynamic generation cap from its effective
    // prompt length.
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
    }
    tracing::trace!("Received responses request: {:?}", request.inner);

    // Resolve an alias to its primary served name and rewrite the request so
    // engine routing, metrics, and the response model all use the canonical
    // primary. The Responses API wraps model in Option<String>, so re-wrap
    // after resolution. Non-aliases pass through metric_model_for's guard.
    let original_model = request.inner.model.clone().unwrap_or_default();
    let canonical = state.manager().resolve_canonical_name(&original_model);
    if canonical != original_model {
        request.inner.model = Some(canonical.clone());
    }
    let model = canonical;
    let streaming = request.inner.stream.unwrap_or(false);
    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
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
        parallel_tool_calls: request.inner.parallel_tool_calls,
        store: request.inner.store,
        tools: request.inner.tools.clone(),
        tool_choice: request.inner.tool_choice.clone(),
        instructions: request.inner.instructions.clone(),
        reasoning: request.inner.reasoning.clone(),
        text: request.inner.text.clone(),
        service_tier: request.inner.service_tier,
        include: request.inner.include.clone(),
        truncation: request.inner.truncation,
        // Upstream `CreateResponse` doesn't carry these yet; plumbed through so
        // the response serializer can default to 0.0 without hardcoding at the
        // build site. When upstream (or our shadow) adds the fields, sourcing
        // from the request becomes a one-line change here.
        presence_penalty: None,
        frequency_penalty: None,
        // Pass-through metadata — accepted on the request, echoed back on the
        // response so the caller can confirm receipt. Dynamo doesn't act on
        // these; see `validate_response_unsupported_fields` for rationale.
        prompt_cache_key: request.inner.prompt_cache_key.clone(),
        prompt_cache_retention: request.inner.prompt_cache_retention,
        safety_identifier: request.inner.safety_identifier.clone(),
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
    if let Err(err_response) = normalize_chat_reasoning_template_args(&mut chat_request) {
        inflight_guard.mark_error(extract_error_type_from_response(&err_response));
        return Err(err_response);
    }

    // Always use internal streaming for aggregation.
    // Set stream_options.include_usage so the backend sends token counts in the final chunk.
    chat_request.inner.stream = Some(true);
    chat_request.inner.stream_options =
        Some(dynamo_protocols::types::ChatCompletionStreamOptions {
            include_usage: true,
            continuous_usage_stats: false,
        });

    let mut request = context.map(|mut _req| chat_request);
    if response_params.max_output_tokens.is_none() {
        request.insert(PRESERVE_OMITTED_MAX_TOKENS_CONTEXT_KEY, true);
    }

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|e| {
            let err_response = ErrorMessage::from_model_error(&e);
            inflight_guard.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    // Gate the experimental v2 batch finalize on the request's tool_choice, mirroring the
    // streaming gate (required/named + structural-tag stay on the v1 finalize path).
    let parsing_options = parsing_options.with_experimental_v2_batch_eligible(
        crate::protocols::openai::chat_completions::tool_parser_v2::batch_tool_choice_eligible(
            request.inner.tool_choice.as_ref(),
        ),
    );

    let mut response_collector = state
        .metrics_clone()
        .create_response_collector(&metric_model);

    tracing::trace!("Issuing generate call for responses");

    // issue the generate call on the engine
    let engine_stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::Responses);
        }
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

        let mut converter = match responses_ctx {
            Some(ctx) => ResponseStreamConverter::with_context(model.clone(), response_params, ctx),
            None => ResponseStreamConverter::new(model.clone(), response_params),
        };

        let mut http_queue_guard = Some(http_queue_guard);

        let mut engine_stream = Box::pin(engine_stream);
        let full_stream = async_stream::stream! {
            let mut events = Vec::with_capacity(4);
            converter.append_start_events(&mut events);
            for event in events.drain(..) {
                yield event.map_err(axum::Error::new);
            }

            // Track whether the backend sent an error event during the stream.
            let mut saw_error = false;

            while let Some(annotated_chunk) = engine_stream.next().await {
                process_chat_response_and_observe_metrics(
                    &annotated_chunk,
                    &mut response_collector,
                    &mut http_queue_guard,
                );

                if extract_backend_error_if_present(&annotated_chunk).is_some() {
                    saw_error = true;
                    continue;
                }

                let Some(stream_resp) = annotated_chunk.data else {
                    continue;
                };

                converter.append_chunk_events(&stream_resp, &mut events);
                for event in events.drain(..) {
                    yield event.map_err(axum::Error::new);
                }
            }

            if saw_error {
                converter.append_error_events(&mut events);
            } else {
                converter.append_end_events(&mut events);
            }
            for event in events.drain(..) {
                yield event.map_err(axum::Error::new);
            }
        };

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
            process_chat_response_and_observe_metrics(
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
                    let err_response =
                        ErrorMessage::internal_server_error("Failed to fold responses stream");
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

    if let Some(field) = request
        .nvext
        .as_ref()
        .and_then(|nvext| nvext.extra_fields.as_ref())
        .and_then(|fields| {
            fields
                .iter()
                .find(|field| matches!(field.as_str(), "completion_token_ids" | "prompt_logprobs"))
        })
    {
        return Some(ErrorMessage::not_implemented_error(format!(
            "{VALIDATION_PREFIX}`nvext.extra_fields=[\"{field}\"]` is not supported by the Responses API."
        )));
    }

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
    // Reject directive fields that change semantics if silently dropped.
    // `max_tool_calls` is a hard cap on tool invocations — accepting it
    // without enforcement would let a caller send `max_tool_calls: 5` and
    // see `max_tool_calls: null` in the response, assuming their limit was
    // honored. Fail loud until real enforcement lands.
    //
    // Pass-through metadata fields (`prompt_cache_key`,
    // `prompt_cache_retention`, `safety_identifier`) are deliberately
    // accepted and echoed back on the response instead. They're hints for
    // OpenAI's caching/moderation backends, not directives — Codex sends
    // `prompt_cache_key` on every request — and the OpenResponses spec
    // includes them on the response body, so echoing the caller's value
    // makes receipt observable without needing a real backend.
    if inner.max_tool_calls.is_some() {
        return Some(ErrorMessage::not_implemented_error(
            VALIDATION_PREFIX.to_string() + "`max_tool_calls` is not supported.",
        ));
    }
    None
}

// todo - abstract this to the top level lib.rs to be reused
pub(crate) fn check_ready(state: &Arc<service_v2::State>) -> Result<(), ErrorResponse> {
    if !state.is_ready() {
        return Err(ErrorMessage::_service_unavailable());
    }
    Ok(())
}

/// Canonical, customer-facing message for "model is registered but not yet
/// ready to serve requests" (deployment still initializing or incomplete).
///
/// One message for every not-ready cause — whichever worker role is missing,
/// the client sees the same text. Deliberately free of internal taxonomy
/// (worker types, namespaces, "worker set"): it stays clear and actionable for
/// end users without leaking deployment internals. Operators get the detailed,
/// per-role breakdown from `GET /v1/models/{model}/ready` instead.
pub(crate) fn model_not_ready_message(model_name: &str) -> String {
    format!(
        "Model `{model_name}` is not ready to serve requests yet. \
         The deployment may still be starting up or is not fully provisioned. \
         Please retry shortly."
    )
}

/// Per-model serving readiness gate.
///
/// Composes AND-wise with [`check_ready`]: a request is admitted only when
/// (a) the process is ready (`check_ready`) AND (b) at least one namespace
/// for this specific model has a complete set of workers — every worker's
/// `needs` DNF is satisfied by the worker types currently present in that
/// namespace.
///
/// Returns `503 Service Unavailable` with a structured body when the model
/// isn't ready to serve. Models the frontend has never heard of fall through
/// here; the per-handler engine lookup later in the request path returns a
/// 404 instead, which is the right shape for "unknown model".
pub(crate) fn check_model_serving_ready(
    state: &Arc<service_v2::State>,
    model_name: &str,
) -> Result<(), ErrorResponse> {
    let Some(model) = state.manager().get_model(model_name) else {
        // Unknown model — let the per-endpoint engine accessor produce the
        // canonical 404. The readiness gate has nothing to say.
        return Ok(());
    };
    if model.has_ready_workers() {
        return Ok(());
    }
    Err(ErrorMessage::service_unavailable_with_body(
        model_not_ready_message(model_name),
    ))
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

    // Build context_length lookup from model deployment cards
    let cards = state.manager().get_model_cards();
    let card_map: HashMap<String, u32> = cards
        .iter()
        .map(|c| (c.display_name.clone(), c.effective_context_length()))
        .collect();

    // Env var overrides (take precedence over MDC values)
    let cw_override: Option<u64> = std::env::var("DYN_CONTEXT_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok());
    let mot_override: Option<u64> = std::env::var("DYN_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok());

    let mut data = Vec::new();

    // Only list models whose worker set is complete in at least one namespace.
    // A registered-but-broken deployment (e.g. decode-only with no prefill peer)
    // is hidden until a peer joins.
    let models: HashSet<String> = state.manager().serving_ready_display_names();
    for model_name in models {
        // Alias entries have no card of their own (keyed by the primary's
        // display_name); fall back to the primary's context length.
        let context_window = cw_override.or_else(|| {
            card_map
                .get(&model_name)
                .or_else(|| card_map.get(&state.manager().resolve_canonical_name(&model_name)))
                .map(|&cl| cl as u64)
        });
        data.push(ModelListing {
            id: model_name.clone(),
            object: "model",
            created,
            owned_by: "nvidia".to_string(),
            context_window,
            max_output_tokens: mot_override,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    context_window: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
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
    let retrieve_path = format!("{}/{{*model_id}}", openai_path);
    let doc_for_openai = RouteDoc::new(axum::http::Method::GET, &openai_path);
    let doc_for_retrieve = RouteDoc::new(axum::http::Method::GET, &retrieve_path);
    // Doc-only: the readiness sub-resource is served by `get_model_openai` via
    // the catch-all retrieve route above (a wildcard must be the terminal
    // segment, so it can't be its own axum route). Advertised for discovery.
    let doc_for_readiness = RouteDoc::new(
        axum::http::Method::GET,
        format!("{}/{{model_id}}/ready", openai_path),
    );

    let router = Router::new()
        .route(&openai_path, get(list_models_openai))
        .route(&retrieve_path, get(get_model_openai))
        .with_state(state);

    (
        vec![doc_for_openai, doc_for_retrieve, doc_for_readiness],
        router,
    )
}

/// Retrieve a single model by ID (OpenAI format).
///
/// Per the OpenAI API spec: `GET /v1/models/{model}` returns a model object.
/// Uses wildcard path to support model IDs with slashes (e.g. `Qwen/Qwen3.5-35B-A3B-FP8`).
async fn get_model_openai(
    State(state): State<Arc<service_v2::State>>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;

    let model_id = model_id.strip_prefix('/').unwrap_or(&model_id);

    // The retrieve route (`/v1/models/{*model_id}`) is a catch-all, so model
    // IDs can contain '/' — and may even end in '/ready'. We therefore
    // dispatch by precedence: an *exact* model match always wins, and only when
    // there is no such model do we treat a trailing `/ready` as the
    // per-model readiness sub-resource (Mechanism 4). This means a model
    // literally named `foo/ready` is still retrievable and never shadowed.
    //
    // Exact match is resolved against ALL registered models (`get_model`), not
    // just the displayable ones, so a registered-but-not-yet-ready `foo/ready`
    // still wins over the readiness sub-resource of a sibling `foo`.
    // `get_model_retrieve` applies the readiness gate itself (503 if not ready).
    if state.manager().get_model(model_id).is_some() {
        return get_model_retrieve(&state, model_id);
    }

    // Readiness sub-resource. Resolves against all registered models (above
    // exact check failed, so `model_id` is not itself a registered model);
    // the whole point of this endpoint is to diagnose models that are
    // registered but not yet ready, so it must find them too.
    if let Some(base) = model_id.strip_suffix("/ready")
        && state.manager().get_model(base).is_some()
    {
        return get_model_readiness(&state, base);
    }

    Err(ErrorMessage::model_not_found())
}

/// `GET /v1/models/{model}` — the OpenAI retrieve-model object. Reports the
/// model only if it is ready to serve (mirrors the `list_models_openai` filter).
fn get_model_retrieve(
    state: &Arc<service_v2::State>,
    model_id: &str,
) -> Result<Response, ErrorResponse> {
    check_model_serving_ready(state, model_id)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Alias entries have no card of their own (cards are keyed by the primary's
    // display_name); fall back to the primary so an alias reports the same
    // context_window that `GET /v1/models` lists for it.
    let canonical_model = state.manager().resolve_canonical_name(model_id);
    let cards = state.manager().get_model_cards();
    let context_length = cards
        .iter()
        .find(|c| c.display_name == model_id || c.display_name == canonical_model)
        .map(|c| c.effective_context_length() as u64);
    let context_window: Option<u64> = std::env::var("DYN_CONTEXT_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .or(context_length);
    let max_output_tokens: Option<u64> = std::env::var("DYN_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok());

    Ok(Json(ModelListing {
        id: model_id.to_string(),
        object: "model",
        created,
        owned_by: "nvidia".to_string(),
        context_window,
        max_output_tokens,
    })
    .into_response())
}

/// `GET /v1/models/{model}/ready` — structured per-namespace worker readiness
/// detail (Mechanism 4). Deliberately *not* readiness-gated: it exists to
/// diagnose models that are not yet ready, so it returns 200 with the full
/// breakdown regardless of whether the model would be served.
fn get_model_readiness(
    state: &Arc<service_v2::State>,
    model_id: &str,
) -> Result<Response, ErrorResponse> {
    let model = state
        .manager()
        .get_model(model_id)
        .ok_or_else(ErrorMessage::model_not_found)?;
    Ok(Json(model.namespace_readiness()).into_response())
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
    // (per-model readiness check is deferred until after we resolve the
    // ImageModel enum into a string; see below)
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
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
            dynamo_protocols::types::ImageModel::GptImage1 => "gpt-image-1".to_string(),
            dynamo_protocols::types::ImageModel::GptImage1dot5 => "gpt-image-1.5".to_string(),
            dynamo_protocols::types::ImageModel::GptImage1Mini => "gpt-image-1-mini".to_string(),
            dynamo_protocols::types::ImageModel::Other(s) => s.clone(),
        })
        .unwrap_or_else(|| "diffusion".to_string());

    // Per-model serving readiness gate (now that we have a resolved model
    // name string).
    check_model_serving_ready(&state, &model)?;

    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    // Get the image generation engine
    let engine = state
        .manager()
        .get_images_engine(&model)
        .map_err(|e| ErrorMessage::from_model_error(&e))?;

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
    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::Images);
        }
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate images");
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

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
            let err_response = ErrorMessage::internal_server_error("Failed to fold images stream");
            inflight.mark_error(extract_error_type_from_response(&err_response));
            err_response
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// Handler for `/v1/images/edits` (I2I). Requires `input_reference`.
async fn images_edits(
    state: State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateImageRequest>,
) -> Result<Response, ErrorResponse> {
    if request.input_reference.is_none() {
        let code = StatusCode::BAD_REQUEST;
        return Err((
            code,
            Json(ErrorMessage {
                message: "input_reference is required for /v1/images/edits".to_string(),
                error_type: map_error_code_to_error_type(code),
                code: code.as_u16(),
                details: None,
            }),
        ));
    }
    images(state, headers, Json(request)).await
}

/// Create an Axum [`Router`] for the OpenAI API Images endpoints.
/// `/v1/images/generations` accepts optional `input_reference` (T2I or TI2I).
/// `/v1/images/edits` requires `input_reference` (I2I).
pub fn images_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let generations_path = path.unwrap_or("/v1/images/generations".to_string());
    let edits_path = generations_path.replace("/generations", "/edits");
    let doc = RouteDoc::new(axum::http::Method::POST, &generations_path);
    let edits_doc = RouteDoc::new(axum::http::Method::POST, &edits_path);
    let router = Router::new()
        .route(&generations_path, post(images))
        .route(&edits_path, post(images_edits))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state(state);
    (vec![doc, edits_doc], router)
}

async fn videos(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<NvCreateVideoRequest>,
) -> Result<Response, ErrorResponse> {
    // return a 503 if the service or model is not ready
    check_ready(&state)?;
    check_model_serving_ready(&state, &request.model)?;

    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
    let request_id = request.id().to_string();

    let streaming = request.stream.unwrap_or(false);

    // Get the model name from the request (video generation model)
    let model = request.model.clone();
    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Create http_queue_guard early - tracks time waiting to be processed
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    // Get the video generation engine
    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|e| ErrorMessage::from_model_error(&e))?;

    // this will increment the inflight gauge for the model
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::Videos,
        streaming,
        &request_id,
    );

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // issue the generate call on the engine
    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::Videos);
        }
        let err_response = ErrorMessage::from_anyhow(e, "Failed to generate videos");
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

    let mut http_queue_guard = Some(http_queue_guard);

    if streaming {
        // [gluo TODO] revisit the cancellation handling here,
        // should be unified with chat_completions.
        let ctx = stream.context();
        let (mut connection_handle, stream_handle) = create_connection_monitor(
            ctx.clone(),
            Some(state.metrics_clone()),
            CancellationLabels {
                model: model.clone(),
                endpoint: Endpoint::Videos.to_string(),
                request_type: "stream".to_string(),
            },
        )
        .await;
        let stream = stream.flat_map(move |response| {
            let sse_result = process_response_using_event_converter_and_observe_metrics(
                EventConverter::from(response),
                &mut response_collector,
                &mut http_queue_guard,
            );
            match sse_result {
                Ok(Some(ev)) => stream::iter(vec![Ok(ev)]),
                Ok(None) => stream::iter(vec![]),
                Err(e) => stream::iter(vec![Err(e)]),
            }
        });
        // monitor_for_disconnects: arms stream_handle, pre-marks inflight Cancelled,
        // emits data:[DONE] on natural end, demotes to Internal on mid-stream Err,
        // and kills the engine context when the client disconnects.
        let stream = monitor_for_disconnects(stream, ctx, inflight, stream_handle);

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }
        // Disarm immediately: we return the body directly, so disconnect detection
        // transfers to stream_handle (armed inside monitor_for_disconnects).
        connection_handle.disarm();
        Ok(sse_stream.into_response())
    } else {
        let stream = stream.inspect(move |response| {
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let response = NvVideosResponse::from_annotated_stream(stream)
            .await
            .map_err(|e| {
                tracing::error!("Failed to fold videos stream for {}: {:?}", request_id, e);
                let err_response =
                    ErrorMessage::internal_server_error("Failed to fold videos stream");
                inflight.mark_error(extract_error_type_from_response(&err_response));
                err_response
            })?;

        inflight.mark_ok();
        Ok(Json(response).into_response())
    }
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
    check_model_serving_ready(&state, &request.model)?;

    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
    let model = request.model.clone();
    let metric_model = state.manager().metric_model_for(&model).to_string();

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    let engine = state
        .manager()
        .get_videos_engine(&model)
        .map_err(|e| ErrorMessage::from_model_error(&e))?;

    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Videos, true, request.id());

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::Videos);
        }
        let err_response = ErrorMessage::from_anyhow(e, "Failed to start video stream");
        inflight.mark_error(extract_error_type_from_response(&err_response));
        err_response
    })?;

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
                    inflight.mark_error(ErrorType::Cancelled);
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
            // inflight is already owned by the monitored_stream which handles
            // mark_ok (stream end) and mark_error (cancellation).
            ErrorMessage::internal_server_error_with_details(
                "Failed to build MJPEG response",
                format!("{e}"),
            )
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
    // (per-model readiness check is deferred until after we resolve the
    // Option<String> model field; see below)
    check_ready(&state)?;

    let request_id = get_or_create_request_id(&headers);
    let request = context_from_headers(request, request_id, &headers)?;
    let request_id = request.id().to_string();

    let streaming = false;

    // model is optional in the request; fall back to a model that can actually
    // serve right now (complete worker set), not just any displayable one, so
    // an incomplete deployment doesn't get picked as the implicit default while
    // a ready model exists.
    let model = request.model.clone().unwrap_or_else(|| {
        state
            .manager()
            .serving_ready_display_names()
            .into_iter()
            .next()
            .unwrap_or_default()
    });
    let metric_model = state.manager().metric_model_for(&model).to_string();

    // Per-model serving readiness gate (now that we have a resolved model
    // name string).
    check_model_serving_ready(&state, &model)?;

    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&metric_model);

    let engine = state
        .manager()
        .get_audios_engine(&model)
        .map_err(|e| ErrorMessage::from_model_error(&e))?;

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

    // If b64_json is present (data_source defaulted or explicitly "b64_json"),
    // decode and return binary with content-type from AudioData.output_format.
    // (matching OpenAI/vLLM-Omni behavior: curl --output file.wav)
    if let Some(first) = response.data.first()
        && let Some(b64) = &first.b64_json
        && let Ok(audio_bytes) = base64::engine::general_purpose::STANDARD.decode(b64)
    {
        let content_type = match first.output_format.as_str() {
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
    use crate::protocols::common::extensions::NvExt;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use crate::protocols::openai::common_ext::CommonExt;
    use crate::protocols::openai::completions::NvCreateCompletionRequest;
    use crate::protocols::openai::responses::NvCreateResponse;
    use dynamo_protocols::types::responses::{CreateResponse, Input, PromptConfig};
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
        CreateCompletionRequest, Prompt,
    };

    const BACKUP_ERROR_MESSAGE: &str = "Failed to generate completions";

    #[test]
    fn test_is_json_content_type() {
        assert!(is_json_content_type("application/json"));
        assert!(is_json_content_type("application/json; charset=utf-8"));
        assert!(is_json_content_type("Application/JSON"));
        assert!(is_json_content_type("application/vnd.dynamo+json"));
        assert!(!is_json_content_type("text/plain"));
        assert!(!is_json_content_type("application/json-patch"));
        assert!(!is_json_content_type("application"));
    }

    #[test]
    fn test_ensure_json_content_type_rejects_missing_or_non_json() {
        let headers = HeaderMap::new();
        let err = ensure_json_content_type(&headers).expect_err("missing content type should fail");
        assert_eq!(err.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);

        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::CONTENT_TYPE,
            "text/plain".parse().unwrap(),
        );
        let err =
            ensure_json_content_type(&headers).expect_err("non-json content type should fail");
        assert_eq!(err.0, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[test]
    fn test_parse_chat_completion_request_escapes_control_chars_in_strings() {
        let body = b"{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"log \x1b[33mPK\x03\x04\"}]}";

        let request: NvCreateChatCompletionRequest =
            parse_json_request("chat completions", body).expect("request should parse");

        let message = request
            .inner
            .messages
            .first()
            .expect("message should exist");
        let ChatCompletionRequestMessage::User(user_message) = message else {
            panic!("expected user message");
        };
        let ChatCompletionRequestUserMessageContent::Text(content) = &user_message.content else {
            panic!("expected text content");
        };
        assert_eq!(content, "log \u{1b}[33mPK\u{3}\u{4}");
    }

    #[test]
    fn test_parse_chat_completion_request_replaces_invalid_utf8_in_strings() {
        let body = b"{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"raw \xff data\"}]}";

        let request: NvCreateChatCompletionRequest =
            parse_json_request("chat completions", body).expect("request should parse");

        let message = request
            .inner
            .messages
            .first()
            .expect("message should exist");
        let ChatCompletionRequestMessage::User(user_message) = message else {
            panic!("expected user message");
        };
        let ChatCompletionRequestUserMessageContent::Text(content) = &user_message.content else {
            panic!("expected text content");
        };
        assert_eq!(content, "raw \u{fffd} data");
    }

    #[test]
    fn test_parse_chat_completion_request_escapes_control_char_after_backslash() {
        let body = b"{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"slash \\\nnext\"}]}";

        let request: NvCreateChatCompletionRequest =
            parse_json_request("chat completions", body).expect("request should parse");

        let message = request
            .inner
            .messages
            .first()
            .expect("message should exist");
        let ChatCompletionRequestMessage::User(user_message) = message else {
            panic!("expected user message");
        };
        let ChatCompletionRequestUserMessageContent::Text(content) = &user_message.content else {
            panic!("expected text content");
        };
        assert_eq!(content, "slash \\\nnext");
    }

    #[test]
    fn test_parse_chat_completion_request_keeps_schema_errors() {
        let body = br#"{"model":"test-model","messages":[{"role":"assistant","content":[{"type":"thinking","thinking":"working"}]}]}"#;

        let err =
            match parse_json_request::<NvCreateChatCompletionRequest>("chat completions", body) {
                Ok(_) => panic!("schema should still fail"),
                Err(err) => err,
            };

        assert_eq!(err.0, StatusCode::BAD_REQUEST);
        assert!(
            err.1
                .message
                .contains("ChatCompletionRequestAssistantMessageContent"),
            "unexpected error: {}",
            err.1.message
        );
    }

    #[test]
    fn test_parse_completion_request_escapes_control_chars_in_prompt() {
        let body =
            b"{\"model\":\"test-model\",\"prompt\":\"log \x1b[33mPK\x03\x04\",\"max_tokens\":1}";

        let request: NvCreateCompletionRequest =
            parse_json_request("completions", body).expect("request should parse");

        let Prompt::String(prompt) = &request.inner.prompt else {
            panic!("expected string prompt");
        };
        assert_eq!(prompt, "log \u{1b}[33mPK\u{3}\u{4}");
    }

    #[test]
    fn test_parse_completion_request_replaces_invalid_utf8_in_prompt() {
        let body = b"{\"model\":\"test-model\",\"prompt\":\"raw \xff data\",\"max_tokens\":1}";

        let request: NvCreateCompletionRequest =
            parse_json_request("completions", body).expect("request should parse");

        let Prompt::String(prompt) = &request.inner.prompt else {
            panic!("expected string prompt");
        };
        assert_eq!(prompt, "raw \u{fffd} data");
    }

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
    fn test_openai_nvext_rejects_agent_context() {
        let err = serde_json::from_value::<NvExt>(serde_json::json!({
            "agent_context": {
                "session_id": "run-123"
            }
        }))
        .unwrap_err();

        assert!(err.to_string().contains("unknown field `agent_context`"));
    }

    #[test]
    fn test_copy_context_metadata_preserves_agent_context() {
        let mut source = Context::new(());
        source.insert(
            AGENT_CONTEXT_CONTEXT_KEY,
            AgentContext {
                session_id: "session-123".to_string(),
                parent_session_id: Some("parent-456".to_string()),
                session_final: Some(true),
                kv_hints: None,
            },
        );

        let mut target = Context::new(());
        copy_context_metadata(&source, &mut target);

        let agent_context = target
            .get::<AgentContext>(AGENT_CONTEXT_CONTEXT_KEY)
            .expect("agent context copied");
        assert_eq!(agent_context.session_id, "session-123");
        assert_eq!(
            agent_context.parent_session_id.as_deref(),
            Some("parent-456")
        );
        assert_eq!(agent_context.session_final, Some(true));
    }

    #[test]
    fn test_context_metadata_preserves_session_affinity() {
        let mut headers = HeaderMap::new();
        headers.insert("x-dynamo-session-id", "session-123".parse().unwrap());
        let source = context_from_headers((), "request-1".to_string(), &headers).unwrap();
        let affinity = source
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .expect("session affinity attached");
        assert_eq!(affinity.as_str(), "session-123");

        let mut target = Context::new(());
        copy_context_metadata(&source, &mut target);
        let affinity = target
            .get::<SessionAffinityId>(SESSION_AFFINITY_CONTEXT_KEY)
            .expect("session affinity copied");
        assert_eq!(affinity.as_str(), "session-123");
    }

    #[test]
    fn test_http_error_response_from_anyhow() {
        let err = http_error_from_engine(400).unwrap_err();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::BAD_REQUEST);
        assert_eq!(response.1.message, "custom error message");
    }

    #[test]
    fn test_check_ready_rejects_draining_service() {
        let service = service_v2::HttpService::builder().build().unwrap();
        let state = service.state_clone();

        assert!(check_ready(&state).is_ok());

        state.start_draining();
        let response = check_ready(&state).unwrap_err();
        assert_eq!(response.0, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_error_response_from_anyhow_out_of_range() {
        // Backend-supplied messages outside the 4xx range must NOT be
        // forwarded to the client — they may include internal paths.
        for code in [399u16, 500, 501] {
            let err = http_error_from_engine(code).unwrap_err();
            let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
            assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(response.1.message, "Internal server error");
            assert!(
                !response.1.message.contains("custom error message"),
                "client response must not include the backend-supplied HttpError message"
            );
        }
    }

    #[test]
    fn test_from_http_error_sanitizes_499_message() {
        // Backend may construct HttpError { code: 499, message: "..." }; that
        // message can carry context IDs / queue paths and must not leak.
        let err = HttpError {
            code: 499,
            message: "session abc-123 cancelled at /srv/queue.py:42".to_string(),
        };
        let response = ErrorMessage::from_http_error(err);
        assert_eq!(response.0.as_u16(), 499);
        assert_eq!(response.1.code, 499);
        assert_eq!(response.1.message, "Request cancelled");
        assert!(!response.1.message.contains("abc-123"));
        assert!(!response.1.message.contains("/srv/queue.py"));
    }

    #[test]
    fn test_other_error_response_from_anyhow() {
        // Non-HttpError anyhow chains must NOT be exposed to the client; only
        // the static backup message should appear in the response.
        let err = other_error_from_engine().unwrap_err();
        let leaked_chain = format!("{err:#}");
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.1.message, BACKUP_ERROR_MESSAGE);
        assert!(
            !response.1.message.contains(&leaked_chain),
            "client response must not contain the anyhow error chain"
        );
    }

    #[test]
    fn test_resource_exhausted_error_response_from_anyhow() {
        use dynamo_runtime::error::{DynamoError, ErrorType};
        use dynamo_runtime::pipeline::error::PipelineError;

        let cause = PipelineError::ServiceOverloaded(
            "All workers are busy, please retry later".to_string(),
        );
        let err: anyhow::Error = DynamoError::builder()
            .error_type(ErrorType::ResourceExhausted)
            .message("All workers are busy, please retry later")
            .cause(cause)
            .build()
            .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(response.0.as_u16(), 529);
        assert_eq!(response.1.code, 529);
        assert_eq!(response.1.error_type, "Overloaded");
        assert_eq!(response.1.message, "Service temporarily overloaded");
        assert!(
            !response.1.message.contains("All workers are busy"),
            "client response must not include the underlying engine message"
        );
    }

    #[test]
    fn unavailable_error_response_from_anyhow() {
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let err: anyhow::Error = DynamoError::builder()
            .error_type(ErrorType::Unavailable)
            .message("No workers available for endpoint test/worker/generate")
            .build()
            .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);

        assert_eq!(response.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.1.code, StatusCode::SERVICE_UNAVAILABLE.as_u16());
        assert_eq!(response.1.message, "Service temporarily unavailable");
    }

    #[test]
    fn queue_rejection_maps_to_structured_http_529() {
        use dynamo_kv_router::scheduling::{QueueLimitKind, QueueRejection};

        let rejection = QueueRejection {
            policy_class: "latency".to_string(),
            limit_kind: QueueLimitKind::CachedTokens,
            current: 2048,
            limit: 1024,
        };
        let response =
            ErrorMessage::from_anyhow(anyhow::Error::new(rejection), BACKUP_ERROR_MESSAGE);

        assert_eq!(response.0.as_u16(), 529);
        assert_eq!(response.1.code, 529);
        assert_eq!(response.1.error_type, "Overloaded");
        assert_eq!(
            response.1.details.as_deref(),
            Some(&serde_json::json!({
                "policy_class": "latency",
                "limit_kind": "cached_tokens",
                "current": 2048,
                "limit": 1024,
            }))
        );
    }

    #[test]
    fn test_nested_invalid_argument_response_from_anyhow() {
        use dynamo_runtime::error::{DynamoError, ErrorType};

        #[derive(Debug)]
        struct WrappedError {
            source: DynamoError,
        }

        impl std::fmt::Display for WrappedError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "outer routing failure")
            }
        }

        impl std::error::Error for WrappedError {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.source)
            }
        }

        let source = DynamoError::builder()
            .error_type(ErrorType::InvalidArgument)
            .message(
                "Request payload is too large for this deployment. Reduce the input size or metadata size and retry.",
            )
            .build();
        let err: anyhow::Error = WrappedError { source }.into();

        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);

        assert_eq!(response.0, StatusCode::BAD_REQUEST);
        assert_eq!(response.1.code, StatusCode::BAD_REQUEST.as_u16());
        assert!(response.1.message.contains("Request payload is too large"));
        assert!(!response.1.message.contains("NATS"));
        assert!(!response.1.message.contains("payload_bytes"));
    }

    #[test]
    fn test_backend_invalid_argument_surfaces_as_400() {
        // `Backend(InvalidArgument)` is what `py_err_to_dynamo` produces
        // for Python `ValueError` / `TypeError` raised inside an engine's
        // `generate()` — must map to 400, not 500.
        use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};

        let err: anyhow::Error = DynamoError::builder()
            .error_type(ErrorType::Backend(BackendError::InvalidArgument))
            .message("Dynamo's SGLang backend does not currently support logprobs >= 1")
            .build()
            .into();

        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);

        assert_eq!(response.0, StatusCode::BAD_REQUEST);
        assert_eq!(response.1.code, StatusCode::BAD_REQUEST.as_u16());
        assert!(response.1.message.contains("does not currently support"));
    }

    #[test]
    fn test_cancelled_error_response_from_anyhow() {
        use dynamo_runtime::error::{DynamoError, ErrorType};

        let err: anyhow::Error = DynamoError::builder()
            .error_type(ErrorType::Cancelled)
            .message("Context id abc-123 is stopped or killed")
            .build()
            .into();
        let response = ErrorMessage::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(
            response.0.as_u16(),
            499,
            "Cancelled errors should return HTTP 499"
        );
        assert_eq!(response.1.code, 499);
        assert_eq!(response.1.error_type, "Client Closed Request");
        // The client gets a static message; the backend detail (context id,
        // cancellation internals) must not leak into the 499 body.
        assert_eq!(response.1.message, "Request cancelled");
        assert!(!response.1.message.contains("abc-123"));
        assert!(!response.1.message.contains("stopped or killed"));
    }

    #[test]
    fn test_cancelled_error_metrics_classification() {
        // HTTP 499 should be classified as Cancelled for metrics
        let error_type =
            classify_error_for_metrics(StatusCode::from_u16(499).unwrap(), "cancelled request");
        assert_eq!(
            error_type,
            ErrorType::Cancelled,
            "HTTP 499 should map to ErrorType::Cancelled in metrics"
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
    fn test_validate_unsupported_fields_accepts_store() {
        let mut request = make_base_request();
        request.inner.store = Some(true);
        let result = validate_response_unsupported_fields(&request);
        assert!(
            result.is_none(),
            "store should be supported for audit opt-in"
        );
    }

    #[tokio::test]
    async fn test_validate_unsupported_fields_rejects_rl_nvext_fields() {
        for field in ["completion_token_ids", "prompt_logprobs"] {
            for stream in [false, true] {
                let mut request = make_base_request();
                request.inner.stream = Some(stream);
                request.nvext = Some(
                    NvExt::builder()
                        .extra_fields(vec![field.to_string()])
                        .build()
                        .unwrap(),
                );

                let response = validate_response_unsupported_fields(&request)
                    .expect("RL nvext response field should be rejected")
                    .into_response();
                assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);

                let body = axum::body::to_bytes(response.into_body(), get_body_limit())
                    .await
                    .unwrap();
                let error: ErrorMessage = serde_json::from_slice(&body).unwrap();
                assert_eq!(
                    error.message,
                    format!(
                        "{VALIDATION_PREFIX}`nvext.extra_fields=[\"{field}\"]` is not supported by the Responses API."
                    )
                );
            }
        }
    }

    #[test]
    fn test_validate_unsupported_fields_rejects_mixed_nvext_fields() {
        let mut request = make_base_request();
        request.nvext = Some(
            NvExt::builder()
                .extra_fields(vec![
                    "timing".to_string(),
                    "completion_token_ids".to_string(),
                ])
                .build()
                .unwrap(),
        );

        assert!(validate_response_unsupported_fields(&request).is_some());
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_supported_nvext_fields() {
        let mut request = make_base_request();
        request.nvext = Some(
            NvExt::builder()
                .extra_fields(vec!["timing".to_string(), "worker_id".to_string()])
                .build()
                .unwrap(),
        );

        assert!(validate_response_unsupported_fields(&request).is_none());
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
            ("max_tool_calls", Box::new(|r| r.max_tool_calls = Some(5))),
        ];

        for (field, set_field) in unsupported_cases {
            let mut req = make_base_request();
            (set_field)(&mut req.inner);
            let result = validate_response_unsupported_fields(&req);
            assert!(result.is_some(), "Expected rejection for `{field}`");
        }
    }

    /// Pass-through metadata fields (`prompt_cache_key`,
    /// `prompt_cache_retention`, `safety_identifier`) are accepted at the
    /// validation layer; the response serializer echoes them back so the
    /// caller can confirm receipt. Codex sends `prompt_cache_key` on every
    /// request — rejecting it broke `codex exec` end-to-end.
    #[test]
    fn test_validate_unsupported_fields_accepts_passthrough_metadata() {
        #[allow(clippy::type_complexity)]
        let passthrough_cases: Vec<(&str, Box<dyn FnOnce(&mut CreateResponse)>)> = vec![
            (
                "prompt_cache_key",
                Box::new(|r| r.prompt_cache_key = Some("ck-1".into())),
            ),
            (
                "prompt_cache_retention",
                Box::new(|r| {
                    r.prompt_cache_retention =
                        Some(dynamo_protocols::types::responses::PromptCacheRetention::InMemory)
                }),
            ),
            (
                "safety_identifier",
                Box::new(|r| r.safety_identifier = Some("user-hash".into())),
            ),
        ];

        for (field, set_field) in passthrough_cases {
            let mut req = make_base_request();
            (set_field)(&mut req.inner);
            let result = validate_response_unsupported_fields(&req);
            assert!(
                result.is_none(),
                "Expected `{field}` to be accepted as pass-through metadata"
            );
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        };
        let result = validate_chat_completion_required_fields(&request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalize_chat_reasoning_template_args_error_response() {
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(serde_json::json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "thinking": {"type": "auto"}
            }))
            .unwrap();

        let result = normalize_chat_reasoning_template_args(&mut request);
        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(
                error_response.1.message,
                format!(
                    "{VALIDATION_PREFIX}`thinking.type` must be `enabled`, `disabled`, or `adaptive`"
                )
            );
        }
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
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
            // Backend-supplied 5xx text must not be forwarded to the client.
            assert_eq!(error_response.1.message, "Internal server error");
            assert!(
                !error_response
                    .1
                    .message
                    .contains("Backend service unavailable")
            );
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_typed_invalid_argument() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
        use futures::stream;

        for error_type in [
            ErrorType::InvalidArgument,
            ErrorType::Backend(BackendError::InvalidArgument),
        ] {
            let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
                data: None,
                id: None,
                event: Some("error".to_string()),
                comment: None,
                error: Some(
                    DynamoError::builder()
                        .error_type(error_type)
                        .message("unsupported JSON schema keyword")
                        .build(),
                ),
            };

            let result = check_for_backend_error(stream::iter(vec![error_event])).await;

            let error_response = match result {
                Err(error_response) => error_response,
                Ok(_) => panic!("typed invalid argument must fail"),
            };
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(error_response.1.code, StatusCode::BAD_REQUEST.as_u16());
            assert_eq!(error_response.1.error_type, "Bad Request");
            assert_eq!(error_response.1.message, "unsupported JSON schema keyword");
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
            // 500 backend JSON messages are sanitized to a static client
            // message; the raw payload is only logged server-side.
            assert_eq!(error_response.1.message, "Internal server error");
            assert_eq!(error_response.1.code, 500);
            assert!(!error_response.1.message.contains("prompt > max_seq_len"));
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_non_client_error_code() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // A backend asserting a non-4xx code (here 399) must not be able to
        // smuggle a sensitive message through with a non-error status:
        // anything outside the 4xx range is sanitized to 500.
        let error_json =
            r#"{"message":"panic at /srv/model.py:42","type":"Backend Error","code":399}"#;
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error_json.to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(error_response.1.code, 500);
            assert_eq!(error_response.1.message, "Internal server error");
            assert!(!error_response.1.message.contains("/srv/model.py"));
            assert!(!error_response.1.message.contains("panic"));
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_503_preserves_status() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Backend 5xx status codes must round-trip so clients can distinguish
        // retryable overload (503) from generic 500; only the body is sanitized.
        let error_json = r#"{"message":"engine pool exhausted at /srv/engine.py:88","code":503}"#;
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error_json.to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(error_response.1.code, 503);
            assert_eq!(error_response.1.message, "Internal server error");
            assert!(!error_response.1.message.contains("engine pool"));
            assert!(!error_response.1.message.contains("/srv/engine.py"));
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_with_499_sanitizes_cancellation() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // 499 falls inside is_client_error(); ensure cancellation text from
        // the backend (e.g. context IDs) cannot reach the client.
        let error_json =
            r#"{"message":"Context id abc-123 cancelled at /srv/queue.py:42","code":499}"#;
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error_json.to_string()]),
            error: None,
        };

        let test_stream = stream::iter(vec![error_event]);
        let result = check_for_backend_error(test_stream).await;

        assert!(result.is_err());
        if let Err(error_response) = result {
            assert_eq!(error_response.0.as_u16(), 499);
            assert_eq!(error_response.1.code, 499);
            assert_eq!(error_response.1.message, "Request cancelled");
            assert!(!error_response.1.message.contains("abc-123"));
            assert!(!error_response.1.message.contains("/srv/queue.py"));
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_skips_leading_annotation_frames() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use futures::stream;

        // Streams prepend a request_id annotation before forwarding engine
        // events. An immediate backend error in the next slot must still be
        // caught so a 4xx surfaces as a 4xx instead of falling through to
        // the generic fold/parse 500.
        let annotation = Annotated::<NvCreateChatCompletionStreamResponse>::from_annotation(
            ANNOTATION_REQUEST_ID,
            &"req-123".to_string(),
        )
        .expect("annotation construction should succeed");
        let error_event = Annotated::<NvCreateChatCompletionStreamResponse> {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![
                r#"{"message":"bad input from client","code":400}"#.to_string(),
            ]),
            error: None,
        };

        let test_stream = stream::iter(vec![annotation, error_event]);
        let result = check_for_backend_error(test_stream).await;

        assert!(
            result.is_err(),
            "annotation followed by an error event must still be detected as an error"
        );
        if let Err(error_response) = result {
            assert_eq!(error_response.0, StatusCode::BAD_REQUEST);
            assert_eq!(error_response.1.code, 400);
            assert_eq!(error_response.1.message, "bad input from client");
        }
    }

    #[tokio::test]
    async fn test_check_for_backend_error_replays_leading_annotation_frames() {
        use crate::types::openai::chat_completions::NvCreateChatCompletionStreamResponse;
        use dynamo_protocols::types::CreateChatCompletionStreamResponse;
        use futures::stream::{self, StreamExt};

        // A leading annotation followed by a normal data event must yield
        // a stream that replays both, in their original order.
        let annotation = Annotated::<NvCreateChatCompletionStreamResponse>::from_annotation(
            ANNOTATION_REQUEST_ID,
            &"req-123".to_string(),
        )
        .expect("annotation construction should succeed");
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
                llm_metrics: None,
            }),
            id: Some("msg-1".to_string()),
            event: None,
            comment: None,
            error: None,
        };

        let test_stream = stream::iter(vec![annotation, normal_event]);
        let result = check_for_backend_error(test_stream).await;

        assert!(result.is_ok());
        let mut returned: Vec<_> = result.unwrap().collect().await;
        assert_eq!(returned.len(), 2, "annotation + data event must replay");
        let first = returned.remove(0);
        assert_eq!(first.event.as_deref(), Some(ANNOTATION_REQUEST_ID));
        let second = returned.remove(0);
        assert_eq!(second.id, Some("msg-1".to_string()));
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
                llm_metrics: None,
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
            // Backend comment text falls under the 5xx default — must be
            // sanitized so it cannot leak internals to the client.
            assert_eq!(error_response.1.message, "Internal server error");
            assert!(!error_response.1.message.contains("Connection timeout"));
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
            classify_error_for_metrics(StatusCode::SERVICE_UNAVAILABLE, "Unavailable"),
            ErrorType::Unavailable
        );
        assert_eq!(
            classify_error_for_metrics(overload_status_code(), "Overloaded"),
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
    fn test_extract_error_type_from_response_unavailable() {
        let response =
            ErrorMessage::from_model_error(&ModelManagerError::ModelUnavailable("x".to_string()));
        assert_eq!(
            extract_error_type_from_response(&response),
            ErrorType::Unavailable
        );
    }

    #[test]
    fn test_from_model_error_maps_correctly() {
        let not_found = ModelManagerError::ModelNotFound("x".to_string());
        assert_eq!(
            ErrorMessage::from_model_error(&not_found).0,
            StatusCode::NOT_FOUND
        );

        let unavailable = ModelManagerError::ModelUnavailable("x".to_string());
        assert_eq!(
            ErrorMessage::from_model_error(&unavailable).0,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    /// The not-ready 503 must be customer-facing: clear and actionable, but free
    /// of internal worker-role / topology taxonomy. Whichever role is missing
    /// (prefill or decode), the client sees the same text — so the message must
    /// never name a specific role, namespace, or "worker set".
    #[test]
    fn test_model_not_ready_message_hides_internals() {
        let msg = model_not_ready_message("my-model").to_lowercase();
        for leak in [
            "prefill",
            "decode",
            "encode",
            "worker",
            "namespace",
            "needs",
        ] {
            assert!(
                !msg.contains(leak),
                "not-ready message leaks internal term `{leak}`: {msg}"
            );
        }
        // Still names the model and signals retryability.
        assert!(model_not_ready_message("my-model").contains("my-model"));
        assert!(msg.contains("retry"));
    }

    /// The dispatch-time backstop (`from_model_error` on `ModelUnavailable`) and
    /// the up-front readiness gate must speak with one voice: identical 503 body
    /// for the same "registered but not servable" condition, regardless of which
    /// role (prefill vs decode) is the missing one.
    #[test]
    fn test_unavailable_paths_share_one_message() {
        let backstop = ErrorMessage::from_model_error(&ModelManagerError::ModelUnavailable(
            "my-model".to_string(),
        ));
        assert_eq!(backstop.0, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(backstop.1.message, model_not_ready_message("my-model"));

        // The gate constructs its body from the same canonical helper, so the
        // two paths cannot drift apart.
        let gate = ErrorMessage::service_unavailable_with_body(model_not_ready_message("my-model"));
        assert_eq!(gate.1.message, backstop.1.message);
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
        ChatCompletionStreamResponseDeltaFunctionCall, CreateChatCompletionStreamResponse,
        FinishReason, FunctionCallStream, FunctionType, Role,
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
            llm_metrics: None,
        };
        Annotated {
            id: Some("test-id".to_string()),
            data: Some(response),
            event: None,
            comment: None,
            error: None,
        }
    }

    fn collect_tool_dispatch_events(
        response: &Annotated<NvCreateChatCompletionStreamResponse>,
        dispatched_ids: &mut HashSet<String>,
    ) -> Vec<Result<Event, axum::Error>> {
        let mut events = Vec::new();
        streaming_tool_dispatch_events(response, dispatched_ids, &mut events);
        events
    }

    fn collect_reasoning_dispatch_events(
        response: &Annotated<NvCreateChatCompletionStreamResponse>,
        buffers: &mut HashMap<u32, String>,
    ) -> Vec<Result<Event, axum::Error>> {
        let mut events = Vec::new();
        accumulate_reasoning_dispatch(response, buffers, &mut events);
        events
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
            r#type: Some(FunctionType::Function),
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

        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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

        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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

        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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

        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty(), "should not dispatch without arguments");
    }

    #[test]
    fn test_tool_dispatch_multiple_tool_calls() {
        let tc1 = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_1".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Paris"}"#.to_string()),
            }),
        };
        let tc2 = ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call_2".to_string()),
            r#type: Some(FunctionType::Function),
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
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn test_tool_dispatch_empty_choices() {
        let response = make_stream_response(vec![]);
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
        assert!(events.is_empty());
    }

    #[test]
    fn test_tool_dispatch_mixed_complete_and_incomplete() {
        // One complete tool call and one incomplete (missing arguments = streaming delta).
        // Only the complete one should dispatch.
        let complete = ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_complete".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"Paris"}"#.to_string()),
            }),
        };
        let incomplete = ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call_partial".to_string()),
            r#type: Some(FunctionType::Function),
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
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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
            r#type: Some(FunctionType::Function),
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
            logprobs: None,
        };

        let response = make_stream_response(vec![choice]);
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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

        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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
        let events = collect_tool_dispatch_events(&response, &mut HashSet::new());
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
        let events = collect_tool_dispatch_events(&response, &mut dispatched);
        assert_eq!(events.len(), 1);

        // Second call with same response — should be deduped
        let events = collect_tool_dispatch_events(&response, &mut dispatched);
        assert!(events.is_empty(), "duplicate id should not dispatch twice");
    }

    // ── accumulate_reasoning_dispatch tests ──

    #[test]
    fn test_reasoning_dispatch_accumulates_and_emits_once() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk 1: reasoning token "Let me"
        let r1 = make_stream_response(vec![make_choice_with_reasoning(0, Some("Let me"), None)]);
        let events = collect_reasoning_dispatch_events(&r1, &mut buffers);
        assert!(
            events.is_empty(),
            "should not emit yet — still accumulating"
        );
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Let me"));

        // Chunk 2: reasoning token " think"
        let r2 = make_stream_response(vec![make_choice_with_reasoning(0, Some(" think"), None)]);
        let events = collect_reasoning_dispatch_events(&r2, &mut buffers);
        assert!(
            events.is_empty(),
            "should not emit yet — still accumulating"
        );
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Let me think"));

        // Chunk 3: reasoning ends (None), meaning normal content follows
        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = collect_reasoning_dispatch_events(&r3, &mut buffers);
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
        collect_reasoning_dispatch_events(&r1, &mut buffers);

        // Chunk 2: finish_reason=length while still in reasoning (max_tokens hit)
        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some(" more"),
            Some(FinishReason::Length),
        )]);
        let events = collect_reasoning_dispatch_events(&r2, &mut buffers);
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
        collect_reasoning_dispatch_events(&r1, &mut buffers);

        // Chunk 2: finish_reason=stop while still in reasoning
        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some("."),
            Some(FinishReason::Stop),
        )]);
        let events = collect_reasoning_dispatch_events(&r2, &mut buffers);
        assert_eq!(events.len(), 1, "should flush on FinishReason::Stop");

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Analysis complete.");
    }

    #[test]
    fn test_reasoning_dispatch_no_reasoning_no_event() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk with no reasoning content at all
        let r = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = collect_reasoning_dispatch_events(&r, &mut buffers);
        assert!(events.is_empty(), "no reasoning content = no event");
    }

    #[test]
    fn test_reasoning_dispatch_empty_string_not_accumulated() {
        let mut buffers: HashMap<u32, String> = HashMap::new();

        // Chunk with empty string reasoning (treated as no-reasoning)
        let r = make_stream_response(vec![make_choice_with_reasoning(0, Some(""), None)]);
        let events = collect_reasoning_dispatch_events(&r, &mut buffers);
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
        let events = collect_reasoning_dispatch_events(&response, &mut buffers);
        assert!(events.is_empty());
    }

    #[test]
    fn test_reasoning_dispatch_empty_choices() {
        let mut buffers: HashMap<u32, String> = HashMap::new();
        let response = make_stream_response(vec![]);
        let events = collect_reasoning_dispatch_events(&response, &mut buffers);
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
        let events = collect_reasoning_dispatch_events(&r1, &mut buffers);
        assert!(events.is_empty(), "both still accumulating");
        assert_eq!(buffers.get(&0).map(|s| s.as_str()), Some("Thinking A"));
        assert_eq!(buffers.get(&1).map(|s| s.as_str()), Some("Thinking B"));

        // Choice 0 stops reasoning, choice 1 continues
        let r2 = make_stream_response(vec![
            make_choice_with_reasoning(0, None, None),
            make_choice_with_reasoning(1, Some(" more"), None),
        ]);
        let events = collect_reasoning_dispatch_events(&r2, &mut buffers);
        assert_eq!(events.len(), 1, "only choice 0 should emit");
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "Thinking A");
        assert_eq!(json["index"], 0);

        // Choice 1 stops reasoning
        let r3 = make_stream_response(vec![make_choice_with_reasoning(1, None, None)]);
        let events = collect_reasoning_dispatch_events(&r3, &mut buffers);
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
        collect_reasoning_dispatch_events(&r1, &mut buffers);

        let r2 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = collect_reasoning_dispatch_events(&r2, &mut buffers);
        assert_eq!(events.len(), 1);
        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "First");

        // Second reasoning block — buffer was cleared, should accumulate fresh
        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, Some("Second"), None)]);
        collect_reasoning_dispatch_events(&r3, &mut buffers);

        let r4 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = collect_reasoning_dispatch_events(&r4, &mut buffers);
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
        collect_reasoning_dispatch_events(&r1, &mut buffers);

        let r2 = make_stream_response(vec![make_choice_with_reasoning(
            0,
            Some(" 分析完成 ✅"),
            None,
        )]);
        collect_reasoning_dispatch_events(&r2, &mut buffers);

        let r3 = make_stream_response(vec![make_choice_with_reasoning(0, None, None)]);
        let events = collect_reasoning_dispatch_events(&r3, &mut buffers);
        assert_eq!(events.len(), 1);

        let json = extract_sse_data_json(events[0].as_ref().unwrap());
        assert_eq!(json["reasoning_content"], "让我想想 🤔 分析完成 ✅");
    }

    /// Build a single-choice `NvCreateChatCompletionStreamResponse`.
    #[allow(clippy::too_many_arguments)]
    fn make_delta(
        content: Option<&str>,
        reasoning: Option<&str>,
        tool_calls: Option<Vec<ChatCompletionMessageToolCallChunk>>,
        finish: Option<FinishReason>,
        usage: Option<dynamo_protocols::types::CompletionUsage>,
        role: Option<Role>,
        refusal: Option<&str>,
        function_call: Option<ChatCompletionStreamResponseDeltaFunctionCall>,
    ) -> NvCreateChatCompletionStreamResponse {
        use dynamo_protocols::types::ChatCompletionMessageContent;
        #[allow(deprecated)]
        let choice = ChatChoiceStream {
            index: 0,
            delta: ChatCompletionStreamResponseDelta {
                content: content.map(|s| ChatCompletionMessageContent::Text(s.to_string())),
                function_call,
                tool_calls,
                role,
                refusal: refusal.map(|s| s.to_string()),
                reasoning_content: reasoning.map(|s| s.to_string()),
            },
            finish_reason: finish,
            logprobs: None,
        };
        NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![choice],
                created: 0,
                model: "m".to_string(),
                system_fingerprint: None,
                object: "chat.completion.chunk".to_string(),
                usage,
                service_tier: None,
            },
            nvext: None,
            llm_metrics: None,
        }
    }

    #[test]
    fn test_is_empty_stream_response() {
        // Empty: all-None, no finish, no usage
        assert!(
            is_empty_stream_response(&make_delta(None, None, None, None, None, None, None, None)),
            "all-None delta → empty",
        );

        // Not empty: has content
        assert!(
            !is_empty_stream_response(&make_delta(
                Some("hi"),
                None,
                None,
                None,
                None,
                None,
                None,
                None
            )),
            "content present → not empty",
        );

        // Not empty: has reasoning
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                Some("thinking"),
                None,
                None,
                None,
                None,
                None,
                None
            )),
            "reasoning present → not empty",
        );

        // Not empty: has finish_reason
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                None,
                None,
                Some(FinishReason::Stop),
                None,
                None,
                None,
                None,
            )),
            "finish_reason → not empty",
        );

        // Not empty: has tool_calls
        let tc = vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call_1".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("f".to_string()),
                arguments: Some("{}".to_string()),
            }),
        }];
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                None,
                Some(tc),
                None,
                None,
                None,
                None,
                None
            )),
            "tool_calls present → not empty",
        );

        // Not empty: usage present
        let usage = dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                None,
                None,
                None,
                Some(usage),
                None,
                None,
                None
            )),
            "usage present → not empty",
        );

        // Role-only: still empty (backends repeat role on every chunk)
        assert!(
            is_empty_stream_response(&make_delta(
                None,
                None,
                None,
                None,
                None,
                Some(Role::Assistant),
                None,
                None,
            )),
            "role-only → empty",
        );

        // Not empty: has refusal
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                None,
                None,
                None,
                None,
                None,
                Some("I can't help with that"),
                None,
            )),
            "refusal present → not empty",
        );

        // Not empty: has function_call (deprecated but still in the struct)
        assert!(
            !is_empty_stream_response(&make_delta(
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(ChatCompletionStreamResponseDeltaFunctionCall {
                    name: Some("my_fn".to_string()),
                    arguments: Some("{}".to_string()),
                }),
            )),
            "function_call present → not empty",
        );
    }

    #[test]
    fn test_chat_predicate_filters_text_empty_string() {
        use dynamo_protocols::types::{
            ChatChoiceLogprobs, ChatCompletionMessageContent, ChatCompletionResponseContentPart,
            ChatCompletionResponseContentPartText, ChatCompletionTokenLogprob,
        };

        // `Text("")` arises during multi-byte UTF-8 token assembly and must be
        // filtered, matching `is_empty_completion_stream_response`'s `""` case.
        let resp = make_delta(Some(""), None, None, None, None, None, None, None);
        assert!(
            is_empty_stream_response(&resp),
            "Text(\"\") delta should be filtered as empty",
        );

        // Structurally empty multimodal `Parts(vec![])` is also empty.
        let mut resp = make_delta(None, None, None, None, None, None, None, None);
        resp.inner.choices[0].delta.content = Some(ChatCompletionMessageContent::Parts(Vec::new()));
        assert!(
            is_empty_stream_response(&resp),
            "Parts(vec![]) delta should be filtered as empty",
        );

        // Non-empty multimodal Parts must be preserved.
        let mut resp = make_delta(None, None, None, None, None, None, None, None);
        resp.inner.choices[0].delta.content = Some(ChatCompletionMessageContent::Parts(vec![
            ChatCompletionResponseContentPart::Text(ChatCompletionResponseContentPartText {
                text: "hi".to_string(),
            }),
        ]));
        assert!(
            !is_empty_stream_response(&resp),
            "Parts with content must not be filtered",
        );

        // Empty content alongside a semantic field (finish_reason) must survive.
        let resp = make_delta(
            Some(""),
            None,
            None,
            Some(FinishReason::Stop),
            None,
            None,
            None,
            None,
        );
        assert!(
            !is_empty_stream_response(&resp),
            "Text(\"\") + finish_reason must not be filtered",
        );

        // Per-token logprobs can arrive while content is still `Text("")` during
        // multi-byte assembly; that payload is meaningful and must survive.
        let mut resp = make_delta(Some(""), None, None, None, None, None, None, None);
        resp.inner.choices[0].logprobs = Some(ChatChoiceLogprobs {
            content: Some(vec![ChatCompletionTokenLogprob {
                token: "h".to_string(),
                logprob: -0.5,
                bytes: Some(vec![104]),
                top_logprobs: vec![],
            }]),
            refusal: None,
        });
        assert!(
            !is_empty_stream_response(&resp),
            "Text(\"\") + logprobs must not be filtered",
        );
    }

    // ── completions empty-stream-response tests ──────────────────────

    use dynamo_protocols::types::{Choice, CompletionFinishReason, CreateCompletionResponse};

    /// Build a single-choice `NvCreateCompletionResponse`.
    fn make_completion_chunk(
        text: &str,
        finish: Option<CompletionFinishReason>,
        usage: Option<dynamo_protocols::types::CompletionUsage>,
    ) -> NvCreateCompletionResponse {
        let choice = Choice {
            text: text.to_string(),
            index: 0,
            logprobs: None,
            finish_reason: finish,
        };
        NvCreateCompletionResponse {
            inner: CreateCompletionResponse {
                id: "test".to_string(),
                choices: vec![choice],
                created: 0,
                model: "m".to_string(),
                system_fingerprint: None,
                object: "text_completion".to_string(),
                usage,
            },
            nvext: None,
        }
    }

    #[test]
    fn test_is_empty_completion_stream_response() {
        // Empty: no text, no finish, no usage
        assert!(
            is_empty_completion_stream_response(&make_completion_chunk("", None, None)),
            "empty text, no finish → empty",
        );

        // Not empty: has text
        assert!(
            !is_empty_completion_stream_response(&make_completion_chunk("hi", None, None)),
            "text present → not empty",
        );

        // Not empty: has finish_reason
        assert!(
            !is_empty_completion_stream_response(&make_completion_chunk(
                "",
                Some(CompletionFinishReason::Stop),
                None,
            )),
            "finish_reason → not empty",
        );

        // Not empty: usage present
        let usage = dynamo_protocols::types::CompletionUsage {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        };
        assert!(
            !is_empty_completion_stream_response(&make_completion_chunk("", None, Some(usage))),
            "usage present → not empty",
        );
    }

    // ── decode_base64_embedding_to_floats ────────────────────────────────
    //
    // The Python embedding worker always emits ``embedding`` as a base64
    // string in the new internal wire format; the HTTP handler decodes
    // back to ``Vec<f32>`` at the response boundary when the client
    // requested float. These tests cover the decoder's three invariants:
    // little-endian f32 byte-for-byte equivalence, invalid base64
    // rejection, and non-multiple-of-4 byte length rejection.

    #[test]
    fn decode_base64_embedding_to_floats_round_trips_little_endian_f32() {
        use base64::Engine as _;
        // Avoid 3.14 to side-step ``clippy::approx_constant`` -- the lint
        // would force importing ``std::f32::consts::PI``, which isn't the
        // point of the test.
        let floats: Vec<f32> = vec![0.0, 1.0, -1.0, 2.5, -42.5, f32::MIN, f32::MAX];
        let mut bytes: Vec<u8> = Vec::with_capacity(floats.len() * 4);
        for f in &floats {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let decoded = decode_base64_embedding_to_floats(&encoded)
            .expect("valid base64 of f32 bytes should decode");
        assert_eq!(decoded, floats);
    }

    #[test]
    fn decode_base64_embedding_to_floats_rejects_invalid_base64() {
        // Padding and alphabet violations: standard base64 alphabet is
        // A-Za-z0-9+/= -- the '!' byte forces a decode error.
        let result = decode_base64_embedding_to_floats("not!valid!base64");
        assert!(
            result.is_err(),
            "non-base64 input should fail decode, got Ok({:?})",
            result.ok()
        );
    }

    #[test]
    fn decode_base64_embedding_to_floats_rejects_non_multiple_of_4_byte_length() {
        // 5 raw bytes -> base64 string. The handler must reject because
        // 5 is not a whole number of f32 values.
        use base64::Engine as _;
        let bytes: Vec<u8> = vec![1, 2, 3, 4, 5];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let result = decode_base64_embedding_to_floats(&encoded);
        assert!(result.is_err(), "5-byte payload must fail, got Ok");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("not a multiple of 4"),
            "error should mention the multiple-of-4 check, got: {err_msg}"
        );
    }
}
