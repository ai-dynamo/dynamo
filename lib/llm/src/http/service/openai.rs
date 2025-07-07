// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    pin::Pin,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use dynamo_runtime::pipeline::{AsyncEngineContext, Context};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio_stream::wrappers::ReceiverStream;

use super::{
    error::HttpError,
    metrics::{Endpoint, InflightGuard, ResponseMetricCollector},
    service_v2, RouteDoc,
};
use crate::preprocessor::LLMMetricAnnotation;
use crate::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse},
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    responses::{NvCreateResponse, NvResponse},
};
use crate::request_template::RequestTemplate;
use crate::types::Annotated;

#[derive(Serialize, Deserialize)]
pub(crate) struct ErrorResponse {
    error: String,
}

impl ErrorResponse {
    /// Not Found Error
    pub fn model_not_found() -> (StatusCode, Json<ErrorResponse>) {
        (
            StatusCode::NOT_FOUND,
            Json(ErrorResponse {
                error: "Model not found".to_string(),
            }),
        )
    }

    /// Service Unavailable
    /// This is returned when the service is live, but not ready.
    pub fn _service_unavailable() -> (StatusCode, Json<ErrorResponse>) {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "Service is not ready".to_string(),
            }),
        )
    }

    /// Internal Service Error
    /// Return this error when the service encounters an internal error.
    /// We should return a generic message to the client instead of the real error.
    /// Internal Services errors are the result of misconfiguration or bugs in the service.
    pub fn internal_server_error(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
        tracing::error!("Internal server error: {msg}");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: msg.to_string(),
            }),
        )
    }

    /// Not Implemented Error
    /// Return this error when the client requests a feature that is not yet implemented.
    /// This should be used for features that are planned but not available.
    pub fn not_implemented_error(msg: &str) -> (StatusCode, Json<ErrorResponse>) {
        tracing::error!("Not Implemented error: {msg}");
        (
            StatusCode::NOT_IMPLEMENTED,
            Json(ErrorResponse {
                error: msg.to_string(),
            }),
        )
    }

    /// The OAI endpoints call an [`dynamo.runtime::engine::AsyncEngine`] which are specialized to return
    /// an [`anyhow::Error`]. This method will convert the [`anyhow::Error`] into an [`HttpError`].
    /// If successful, it will return the [`HttpError`] as an [`ErrorResponse::internal_server_error`]
    /// with the details of the error.
    pub fn from_anyhow(err: anyhow::Error, alt_msg: &str) -> (StatusCode, Json<ErrorResponse>) {
        match err.downcast::<HttpError>() {
            Ok(http_error) => ErrorResponse::from_http_error(http_error),
            Err(err) => ErrorResponse::internal_server_error(&format!("{alt_msg}: {err}")),
        }
    }

    /// Implementers should only be able to throw 400-499 errors.
    pub fn from_http_error(err: HttpError) -> (StatusCode, Json<ErrorResponse>) {
        if err.code < 400 || err.code >= 500 {
            return ErrorResponse::internal_server_error(&err.message);
        }
        match StatusCode::from_u16(err.code) {
            Ok(code) => (code, Json(ErrorResponse { error: err.message })),
            Err(_) => ErrorResponse::internal_server_error(&err.message),
        }
    }
}

impl From<HttpError> for ErrorResponse {
    fn from(err: HttpError) -> Self {
        ErrorResponse { error: err.message }
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
#[tracing::instrument(skip_all)]
async fn completions(
    State(state): State<Arc<service_v2::State>>,
    Json(request): Json<NvCreateCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // todo - extract distributed tracing id and context id from headers
    let request_id = uuid::Uuid::new_v4().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // update the request to always stream
    let inner = async_openai::types::CreateCompletionRequest {
        stream: Some(true),
        ..request.inner
    };

    let request = NvCreateCompletionRequest {
        inner,
        nvext: request.nvext,
    };

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_completions_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Completions, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // setup context
    // todo - inherit request_id from distributed trace details
    let request = Context::with_id(request, request_id.clone());

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic

    if streaming {
        let stream = stream.map(move |response| {
            process_event_converter(EventConverter::from(response), &mut response_collector)
        });
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight_guard).await;

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // TODO: report ISL/OSL for non-streaming requests
        let response = NvCreateCompletionResponse::from_annotated_stream(stream.into())
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id,
                    e
                );
                ErrorResponse::internal_server_error("Failed to fold completions stream")
            })?;

        inflight_guard.mark_ok();
        Ok(Json(response).into_response())
    }
}

#[tracing::instrument(skip_all)]
async fn embeddings(
    State(state): State<Arc<service_v2::State>>,
    Json(request): Json<NvCreateEmbeddingRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // todo - extract distributed tracing id and context id from headers
    let request_id = uuid::Uuid::new_v4().to_string();

    // Embeddings are typically not streamed, so we default to non-streaming
    let streaming = false;

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_embeddings_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    // this will increment the inflight gauge for the model
    let mut inflight =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Embeddings, streaming);

    // setup context
    // todo - inherit request_id from distributed trace details
    let request = Context::with_id(request, request_id.clone());

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate embeddings"))?;

    // Embeddings are typically returned as a single response (non-streaming)
    // so we fold the stream into a single response
    let response = NvCreateEmbeddingResponse::from_annotated_stream(stream.into())
        .await
        .map_err(|e| {
            tracing::error!(
                "Failed to fold embeddings stream for {}: {:?}",
                request_id,
                e
            );
            ErrorResponse::internal_server_error("Failed to fold embeddings stream")
        })?;

    inflight.mark_ok();
    Ok(Json(response).into_response())
}

/// OpenAI Chat Completions Request Handler
///
/// This method will handle the incoming request for the /v1/chat/completions endpoint. The endpoint is a "source"
/// for an [`super::OpenAIChatCompletionsStreamingEngine`] and will return a stream of responses which will be
/// forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
#[tracing::instrument(skip_all)]
async fn chat_completions(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    Json(mut request): Json<NvCreateChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

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
    tracing::trace!("Received chat completions request: {:?}", request.inner);

    // todo - extract distributed tracing id and context id from headers
    let request_id = uuid::Uuid::new_v4().to_string();

    // todo - decide on default
    let streaming = request.inner.stream.unwrap_or(false);

    // update the request to always stream
    let inner_request = async_openai::types::CreateChatCompletionRequest {
        stream: Some(true),
        ..request.inner
    };

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        nvext: request.nvext,
    };

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // todo - determine the proper error code for when a request model is not present
    tracing::trace!("Getting chat completions engine for model: {}", model);

    let engine = state
        .manager()
        .get_chat_completions_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::ChatCompletions, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // setup context
    // todo - inherit request_id from distributed trace details
    let request = Context::with_id(request, request_id.clone());

    tracing::trace!("Issuing generate call for chat completions");

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // todo - tap the stream and propagate request level metrics
    // note - we might do this as part of the post processing set to make it more generic

    if streaming {
        let stream = stream.map(move |response| {
            process_event_converter(EventConverter::from(response), &mut response_collector)
        });
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight_guard).await;

        let mut sse_stream = Sse::new(stream);

        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // TODO: report ISL/OSL for non-streaming requests
        let response = NvCreateChatCompletionResponse::from_annotated_stream(stream.into())
            .await
            .map_err(|e| {
                tracing::error!(
                    request_id,
                    "Failed to fold chat completions stream for: {:?}",
                    e
                );
                ErrorResponse::internal_server_error(&format!(
                    "Failed to fold chat completions stream: {}",
                    e
                ))
            })?;

        inflight_guard.mark_ok();
        Ok(Json(response).into_response())
    }
}

/// OpenAI Responses Request Handler
///
/// This method will handle the incoming request for the /v1/responses endpoint.
#[tracing::instrument(skip_all)]
async fn responses(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    Json(mut request): Json<NvCreateResponse>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // return a 503 if the service is not ready
    check_ready(&state)?;

    // Handle unsupported fields - if Some(resp) is returned by validate_unsupported_fields,
    // then a field was used that is unsupported. We will log an error message
    // and early return a 501 NOT_IMPLEMENTED status code. Otherwise, proceeed.
    if let Some(resp) = validate_unsupported_fields(&request) {
        return Ok(resp.into_response());
    }

    // Handle non-text (image, audio, file) inputs - if Some(resp) is returned by
    // validate_input_is_text_only, then we are handling something other than Input::Text(_).
    // We will log an error message and early return a 501 NOT_IMPLEMENTED status code.
    // Otherwise, proceeed.
    if let Some(resp) = validate_input_is_text_only(&request) {
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

    let request_id = uuid::Uuid::new_v4().to_string();

    // Convert NvCreateResponse --> NvCreateChatCompletionRequest
    let request: NvCreateChatCompletionRequest = request.try_into().map_err(|e| {
        tracing::error!(
            request_id,
            "Failed to convert NvCreateResponse to NvCreateChatCompletionRequest: {:?}",
            e
        );
        ErrorResponse::not_implemented_error(&format!(
            "Only Input::Text(_) is currently supported: {}",
            e
        ))
    })?;

    let model = &request.inner.model;

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let engine = state
        .manager()
        .get_chat_completions_engine(model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Responses, false);

    let _response_collector = state.metrics_clone().create_response_collector(model);

    let request = Context::with_id(request, request_id.clone());

    tracing::trace!("Issuing generate call for chat completions");

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to generate completions"))?;

    // TODO: handle streaming, currently just unary
    let response = NvCreateChatCompletionResponse::from_annotated_stream(stream.into())
        .await
        .map_err(|e| {
            tracing::error!(
                request_id,
                "Failed to fold chat completions stream for: {:?}",
                e
            );
            ErrorResponse::internal_server_error(&format!(
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
        ErrorResponse::internal_server_error("Failed to convert internal response")
    })?;

    inflight_guard.mark_ok();

    Ok(Json(response).into_response())
}

pub fn validate_input_is_text_only(request: &NvCreateResponse) -> Option<impl IntoResponse> {
    match &request.inner.input {
        async_openai::types::responses::Input::Text(_) => None,
        _ => Some(ErrorResponse::not_implemented_error("Only `Input::Text` is supported. Structured, multimedia, or custom input types are not yet implemented.")),
    }
}

/// Checks for unsupported fields in the request.
/// Returns Some(response) if unsupported fields are present.
pub fn validate_unsupported_fields(request: &NvCreateResponse) -> Option<impl IntoResponse> {
    let inner = &request.inner;

    if inner.background == Some(true) {
        return Some(ErrorResponse::not_implemented_error(
            "`background: true` is not supported.",
        ));
    }
    if inner.include.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`include` is not supported.",
        ));
    }
    if inner.instructions.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`instructions` is not supported.",
        ));
    }
    if inner.max_tool_calls.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`max_tool_calls` is not supported.",
        ));
    }
    if inner.metadata.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`metadata` is not supported.",
        ));
    }
    if inner.parallel_tool_calls == Some(true) {
        return Some(ErrorResponse::not_implemented_error(
            "`parallel_tool_calls: true` is not supported.",
        ));
    }
    if inner.previous_response_id.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`previous_response_id` is not supported.",
        ));
    }
    if inner.prompt.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`prompt` is not supported.",
        ));
    }
    if inner.reasoning.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`reasoning` is not supported.",
        ));
    }
    if inner.service_tier.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`service_tier` is not supported.",
        ));
    }
    if inner.store == Some(true) {
        return Some(ErrorResponse::not_implemented_error(
            "`store: true` is not supported.",
        ));
    }
    if inner.stream == Some(true) {
        return Some(ErrorResponse::not_implemented_error(
            "`stream: true` is not supported.",
        ));
    }
    if inner.text.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`text` is not supported.",
        ));
    }
    if inner.tool_choice.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`tool_choice` is not supported.",
        ));
    }
    if inner.tools.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`tools` is not supported.",
        ));
    }
    if inner.truncation.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`truncation` is not supported.",
        ));
    }
    if inner.user.is_some() {
        return Some(ErrorResponse::not_implemented_error(
            "`user` is not supported.",
        ));
    }

    None
}

// todo - abstract this to the top level lib.rs to be reused
// todo - move the service_observer to its own state/arc
fn check_ready(_state: &Arc<service_v2::State>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    // if state.service_observer.stage() != ServiceStage::Ready {
    //     return Err(ErrorResponse::service_unavailable());
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
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
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

/// This method will consume a stream of SSE events and forward them to a new stream defined by a tokio channel.
/// In this way, if the downstream is dropped, then the upstream will be unable to send any more events. This is
/// how we can monitor for disconnects and stop the generation of completions.
///
/// If a disconnect is detected, then the context will issue a `stop_generating` call to the context which will
/// propagate the cancellation signal to the backend.
async fn monitor_for_disconnects(
    stream: Pin<
        Box<dyn Stream<Item = Result<axum::response::sse::Event, axum::Error>> + std::marker::Send>,
    >,
    context: Arc<dyn AsyncEngineContext>,
    mut inflight_guard: InflightGuard,
) -> ReceiverStream<Result<Event, axum::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel(8);

    tokio::spawn(async move {
        let mut stream = stream;
        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => Ok(event),
                Err(err) => Ok(Event::default().event("error").comment(err.to_string())),
            };

            if (tx.send(event).await).is_err() {
                tracing::trace!("Forwarding SSE stream was dropped; breaking loop");
                context.stop_generating();
                break;
            }
        }

        // Stream completed successfully - mark as ok
        if tx.send(Ok(Event::default().data("[DONE]"))).await.is_ok() {
            inflight_guard.mark_ok();
        }
    });

    ReceiverStream::new(rx)
}

struct EventConverter<T>(Annotated<T>);

impl<T> From<Annotated<T>> for EventConverter<T> {
    fn from(annotated: Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

fn process_event_converter<T: Serialize>(
    annotated: EventConverter<T>,
    response_collector: &mut ResponseMetricCollector,
) -> Result<Event, axum::Error> {
    let mut annotated = annotated.0;

    // update metrics
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(&annotated) {
        response_collector.observe_current_osl(metrics.output_tokens);
        response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);

        // Chomp the LLMMetricAnnotation so it's not returned in the response stream
        // TODO: add a flag to control what is returned in the SSE stream
        if annotated.event.as_deref() == Some(crate::preprocessor::ANNOTATION_LLM_METRICS) {
            annotated.event = None;
            annotated.comment = None;
        }
    }

    let mut event = Event::default();

    if let Some(data) = annotated.data {
        event = event.json_data(data)?;
    }

    if let Some(msg) = annotated.event {
        if msg == "error" {
            let msgs = annotated
                .comment
                .unwrap_or_else(|| vec!["unspecified error".to_string()]);
            return Err(axum::Error::new(msgs.join(" -- ")));
        }
        event = event.event(msg);
    }

    if let Some(comments) = annotated.comment {
        for comment in comments {
            event = event.comment(comment);
        }
    }

    Ok(event)
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
        .route(&path, post(completions))
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
        .route(&path, post(chat_completions))
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
        .route(&path, post(responses))
        .with_state((state, template));
    (vec![doc], router)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use async_openai::types::responses::{
        CreateResponse, Input, InputContent, InputItem, InputMessage, PromptConfig,
        Role as ResponseRole, ServiceTier, TextConfig, TextResponseFormat, ToolChoice,
        ToolChoiceMode, Truncation,
    };

    use super::*;
    use crate::discovery::ModelManagerError;
    use crate::protocols::openai::responses::NvCreateResponse;

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
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response.error, "custom error message");
    }

    #[test]
    fn test_error_response_from_anyhow_out_of_range() {
        let err = http_error_from_engine(399).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");

        let err = http_error_from_engine(500).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");

        let err = http_error_from_engine(501).unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(response.error, "custom error message");
    }

    #[test]
    fn test_other_error_response_from_anyhow() {
        let err = other_error_from_engine().unwrap_err();
        let (status, response) = ErrorResponse::from_anyhow(err, BACKUP_ERROR_MESSAGE);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            response.error,
            format!(
                "{}: {}",
                BACKUP_ERROR_MESSAGE,
                other_error_from_engine().unwrap_err()
            )
        );
    }

    #[test]
    fn test_validate_input_is_text_only_accepts_text() {
        let request = make_base_request();
        let result = validate_input_is_text_only(&request);
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
        let result = validate_input_is_text_only(&request);
        assert!(result.is_some());
    }

    #[test]
    fn test_validate_unsupported_fields_accepts_clean_request() {
        let request = make_base_request();
        let result = validate_unsupported_fields(&request);
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
            let result = validate_unsupported_fields(&req);
            assert!(result.is_some(), "Expected rejection for `{field}`");
        }
    }
}
