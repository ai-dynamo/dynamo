// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the Anthropic Messages API (`/v1/messages`).
//!
//! This is a translation layer: incoming Anthropic requests are converted to
//! chat completions, processed by the existing engine, and responses/streams
//! are converted back to Anthropic format.

use std::pin::Pin;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{
        IntoResponse, Response,
        sse::{KeepAlive, Sse},
    },
    routing::post,
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{StreamExt, stream};
use tracing::Instrument;

use super::{
    RouteDoc,
    disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects},
    metrics::{Endpoint, process_response_and_observe_metrics},
    service_v2,
};
use crate::preprocessor::OpenAIPreprocessor;
use crate::protocols::anthropic::stream_converter::AnthropicStreamConverter;
use crate::protocols::anthropic::types::{
    AnthropicCountTokensRequest, AnthropicCountTokensResponse, AnthropicCreateMessageRequest,
    AnthropicErrorBody, AnthropicErrorResponse, SystemContent,
    chat_completion_to_anthropic_response,
};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    NvCreateChatCompletionStreamResponse, aggregator::ChatCompletionAggregator,
};
use crate::request_template::RequestTemplate;
use crate::types::Annotated;

// Re-use helpers from the openai module (sibling under service/)
use super::openai::{get_body_limit, get_or_create_request_id};

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Creates the router for the `/v1/messages` and `/v1/messages/count_tokens` endpoints.
pub fn anthropic_messages_router(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/messages".to_string());
    let count_tokens_path = format!("{}/count_tokens", &path);
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let count_doc = RouteDoc::new(axum::http::Method::POST, &count_tokens_path);
    let router = Router::new()
        .route(&path, post(handler_anthropic_messages))
        .route(&count_tokens_path, post(handler_count_tokens))
        .layer(middleware::from_fn(anthropic_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .with_state((state, template));
    (vec![doc, count_doc], router)
}

// ---------------------------------------------------------------------------
// Error middleware
// ---------------------------------------------------------------------------

/// Converts 422 validation errors to Anthropic error format.
async fn anthropic_error_middleware(request: Request<Body>, next: Next) -> Response {
    let response = next.run(request).await;

    if response.status() == StatusCode::UNPROCESSABLE_ENTITY {
        let (_parts, body) = response.into_parts();
        let body_bytes = axum::body::to_bytes(body, get_body_limit())
            .await
            .unwrap_or_default();
        let error_message = String::from_utf8_lossy(&body_bytes).to_string();
        return anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &error_message,
        );
    }

    response
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// Top-level HTTP handler for POST /v1/messages.
async fn handler_anthropic_messages(
    State((state, template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    headers: HeaderMap,
    Json(request): Json<AnthropicCreateMessageRequest>,
) -> Result<Response, Response> {
    // Validate required fields
    if request.messages.is_empty() {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages: field required",
        ));
    }
    if request.max_tokens == 0 {
        return Err(anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "max_tokens: must be greater than 0",
        ));
    }

    // Create request context
    let request_id = get_or_create_request_id(None, &headers);
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // Create connection handles
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context.clone(), Some(state.metrics_clone())).await;

    let response =
        tokio::spawn(anthropic_messages(state, template, request, stream_handle).in_current_span())
            .await
            .map_err(|e| {
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    &format!("Failed to await messages task: {:?}", e),
                )
            })?;

    connection_handle.disarm();
    response
}

/// Core logic for the Anthropic Messages endpoint.
#[tracing::instrument(level = "debug", skip_all, fields(request_id = %request.id()))]
async fn anthropic_messages(
    state: Arc<service_v2::State>,
    template: Option<RequestTemplate>,
    mut request: Context<AnthropicCreateMessageRequest>,
    mut stream_handle: ConnectionHandle,
) -> Result<Response, Response> {
    let streaming = request.stream;
    let request_id = request.id().to_string();

    // Apply template defaults before capturing model (must happen first so
    // engine lookup and metrics use the resolved model name).
    if let Some(template) = template {
        if request.model.is_empty() {
            request.model = template.model.clone();
        }
        if request.temperature.is_none() {
            request.temperature = Some(template.temperature);
        }
        if request.max_tokens == 0 {
            request.max_tokens = template.max_completion_tokens;
        }
    }

    // Strip Claude Code billing preamble from system prompt if enabled
    if env_is_truthy(env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE) {
        strip_billing_preamble(&mut request.system);
    }

    let model = request.model.clone();
    let http_queue_guard = state.metrics_clone().create_http_queue_guard(&model);

    tracing::trace!("Received Anthropic messages request: {:?}", &*request);

    let (orig_request, context) = request.into_parts();
    let model_for_resp = orig_request.model.clone();

    // Check if the Anthropic request explicitly enabled thinking. When thinking
    // is enabled, reasoning-capable models' chat templates typically inject
    // `<think>` into the prompt, so the completion starts mid-reasoning.
    let thinking_enabled = orig_request
        .thinking
        .as_ref()
        .is_some_and(|t| t.thinking_type == "enabled");

    // Convert Anthropic request -> Chat Completion request
    let chat_request: NvCreateChatCompletionRequest =
        orig_request.try_into().map_err(|e: anyhow::Error| {
            tracing::error!(
                request_id,
                error = %e,
                "Failed to convert AnthropicCreateMessageRequest to NvCreateChatCompletionRequest",
            );
            anthropic_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("Failed to convert request: {}", e),
            )
        })?;

    let request = context.map(|_req| chat_request);

    tracing::trace!("Getting chat completions engine for model: {}", model);

    let (engine, parsing_options) = state
        .manager()
        .get_chat_completions_engine_with_parsing(&model)
        .map_err(|_| {
            anthropic_error(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("Model '{}' not found", model),
            )
        })?;

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    tracing::trace!("Issuing generate call for Anthropic messages");

    let engine_stream = engine.generate(request).await.map_err(|e| {
        anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            &format!("Failed to generate completions: {}", e),
        )
    })?;

    let ctx = engine_stream.context();

    // Apply reasoning parser to the engine stream if configured.
    // The preprocessor (which normally handles this for the OpenAI path) is
    // bypassed by the Anthropic endpoint, so we apply the same stream
    // transform here.  This populates `delta.reasoning_content` which the
    // AnthropicStreamConverter translates into thinking content blocks.
    //
    // When thinking is enabled, the model's chat template likely injected
    // `<think>` into the prompt (e.g., Qwen3.5), so the parser must start
    // in reasoning mode — the completion begins mid-reasoning without an
    // explicit `<think>` tag.
    let engine_stream: Pin<
        Box<dyn futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>,
    > = if let Some(ref reasoning_parser_name) = parsing_options.reasoning_parser {
        Box::pin(OpenAIPreprocessor::parse_reasoning_content_from_stream(
            engine_stream,
            reasoning_parser_name.clone(),
            thinking_enabled,
        ))
    } else {
        Box::pin(engine_stream)
    };

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::AnthropicMessages, streaming);

    if streaming {
        stream_handle.arm();

        use std::sync::atomic::{AtomicBool, Ordering};

        let mut converter = AnthropicStreamConverter::new(model_for_resp);
        let start_events = converter.emit_start_events();

        let converter = std::sync::Arc::new(std::sync::Mutex::new(converter));
        let converter_end = converter.clone();

        let saw_error = std::sync::Arc::new(AtomicBool::new(false));
        let saw_error_end = saw_error.clone();

        let mut http_queue_guard = Some(http_queue_guard);

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

        let stream = monitor_for_disconnects(full_stream, ctx, inflight_guard, stream_handle);

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        // Non-streaming path: aggregate stream into single response

        // Check first event for backend errors using the openai helper
        let stream_with_check = super::openai::check_for_backend_error(engine_stream)
            .await
            .map_err(|(status, json_err)| {
                tracing::error!(request_id, %status, ?json_err, "Backend error detected");
                anthropic_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "api_error",
                    "Backend error during generation",
                )
            })?;

        let mut http_queue_guard = Some(http_queue_guard);
        let stream = stream_with_check.inspect(move |response| {
            process_response_and_observe_metrics(
                response,
                &mut response_collector,
                &mut http_queue_guard,
            );
        });

        let chat_response =
            NvCreateChatCompletionResponse::from_annotated_stream(stream, parsing_options.clone())
                .await
                .map_err(|e| {
                    tracing::error!(request_id, "Failed to fold messages stream: {:?}", e);
                    anthropic_error(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "api_error",
                        &format!("Failed to fold messages stream: {}", e),
                    )
                })?;

        let response = chat_completion_to_anthropic_response(chat_response, &model_for_resp);

        inflight_guard.mark_ok();

        Ok(Json(response).into_response())
    }
}

// ---------------------------------------------------------------------------
// Count tokens
// ---------------------------------------------------------------------------

/// Handler for POST /v1/messages/count_tokens.
/// Returns an estimated input token count using a len/3 heuristic.
async fn handler_count_tokens(
    State((_state, _template)): State<(Arc<service_v2::State>, Option<RequestTemplate>)>,
    Json(mut request): Json<AnthropicCountTokensRequest>,
) -> Result<Response, Response> {
    if env_is_truthy(env_llm::DYN_STRIP_ANTHROPIC_PREAMBLE) {
        strip_billing_preamble(&mut request.system);
    }
    let tokens = request.estimate_tokens();
    Ok(Json(AnthropicCountTokensResponse {
        input_tokens: tokens,
    })
    .into_response())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip the Claude Code billing preamble from the system prompt.
///
/// Claude Code prepends `x-anthropic-billing-header: cc_version=...; cch=...;\n`
/// to every system prompt. This varies per session and per release, wasting tokens
/// and preventing prompt prefix caching on the target model.
fn strip_billing_preamble(system: &mut Option<SystemContent>) {
    if let Some(content) = system {
        let trimmed = content.text.trim_start();
        if trimmed.starts_with("x-anthropic-billing-header:")
            && let Some(newline_pos) = trimmed.find('\n')
        {
            content.text = trimmed[newline_pos + 1..].to_string();
        }
    }
}

/// Build an Anthropic-formatted error response.
/// Maps HTTP status codes to Anthropic error types following the Anthropic API spec.
fn anthropic_error(status: StatusCode, error_type: &str, message: &str) -> Response {
    let mapped_type = match status.as_u16() {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        // Use the caller-provided type for other codes (e.g. 500 → "api_error")
        _ => error_type,
    };

    (
        status,
        Json(AnthropicErrorResponse {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: mapped_type.to_string(),
                message: message.to_string(),
            },
        }),
    )
        .into_response()
}
