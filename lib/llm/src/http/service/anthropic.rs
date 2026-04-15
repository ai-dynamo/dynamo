// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP handler for the Anthropic Messages API (`/v1/messages`).
//!
//! This is a translation layer: incoming Anthropic requests are converted to
//! chat completions, processed by the existing engine, and responses/streams
//! are converted back to Anthropic format.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
    routing::{get, post},
};
use dynamo_runtime::config::{env_is_truthy, environment_names::llm as env_llm};
use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
use futures::{StreamExt, stream};
use tracing::Instrument;

use super::{
    RouteDoc,
    disconnect::{ConnectionHandle, create_connection_monitor, monitor_for_disconnects},
    metrics::{CancellationLabels, Endpoint, process_response_and_observe_metrics},
    service_v2,
};
use crate::protocols::anthropic::stream_converter::AnthropicStreamConverter;
use crate::protocols::anthropic::types::{
    AnthropicCountTokensRequest, AnthropicCountTokensResponse, AnthropicCreateMessageRequest,
    AnthropicErrorBody, AnthropicErrorResponse, SystemContent,
    chat_completion_to_anthropic_response,
};
use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse,
    aggregator::ChatCompletionAggregator,
};
use crate::protocols::unified::UnifiedRequest;
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

/// Creates the router for model listing and retrieval.
///
/// When the `anthropic-version` header is present, returns the Anthropic model
/// format (with `context_window`, `display_name`, etc.). Otherwise returns the
/// standard OpenAI format. This keeps Anthropic-specific content negotiation
/// out of the OpenAI handler.
pub fn anthropic_models_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let models_path = path.unwrap_or("/v1/models".to_string());
    let retrieve_path = format!("{}/{{*model_id}}", models_path);
    let list_doc = RouteDoc::new(axum::http::Method::GET, &models_path);
    let retrieve_doc = RouteDoc::new(axum::http::Method::GET, &retrieve_path);
    let router = Router::new()
        .route(&models_path, get(list_models))
        .route(&retrieve_path, get(get_model))
        .with_state(state);
    (vec![list_doc, retrieve_doc], router)
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
    let request_id = get_or_create_request_id(&headers);
    let streaming = request.stream;
    let cancellation_labels = CancellationLabels {
        model: request.model.clone(),
        endpoint: Endpoint::AnthropicMessages.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let request = Context::with_id(request, request_id);
    let context = request.context();

    // Create connection handles
    let (mut connection_handle, stream_handle) = create_connection_monitor(
        context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;

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

    // Look up engine and parsing options early so we know whether a reasoning
    // parser is configured before converting the request.
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

    let (orig_request, context) = request.into_parts();
    let model_for_resp = orig_request.model.clone();

    // Check if the Anthropic request explicitly disabled thinking.
    let thinking_explicitly_disabled = orig_request
        .thinking
        .as_ref()
        .is_some_and(|t| t.thinking_type == "disabled");

    // Estimate input tokens before consuming the request via try_into().
    // Only used in the streaming path to populate message_start.
    let estimated_input_tokens = if streaming {
        estimate_input_tokens(&orig_request)
    } else {
        0
    };

    // Convert Anthropic request -> UnifiedRequest -> Chat Completion request
    let unified_request: UnifiedRequest = orig_request.try_into().map_err(|e: anyhow::Error| {
        tracing::error!(
            request_id,
            error = %e,
            "Failed to convert AnthropicCreateMessageRequest to UnifiedRequest",
        );
        anthropic_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!("Failed to convert request: {}", e),
        )
    })?;

    // Extract the API context before consuming the UnifiedRequest — this
    // carries Anthropic-specific fields (thinking config, cache breakpoints,
    // etc.) that the stream converter needs for faithful response reconstruction.
    let anthropic_ctx = unified_request.anthropic_context().cloned();
    let mut chat_request = unified_request.into_inner();

    // When a reasoning parser is configured and the client hasn't explicitly
    // disabled thinking, assume the model's chat template will inject `<think>`.
    //
    // Two things must be aligned:
    //   1. chat_template_args must include enable_thinking=true so the backend's
    //      template actually injects `<think>` into the prompt. For the
    //      ModelInput::Text path (SGLang without --skip-tokenizer-init), the
    //      backend applies the template — without explicit enable_thinking the
    //      result depends on the template's default which varies by model.
    //   2. prompt_injected_reasoning must be true so the parser starts in
    //      reasoning mode with stripped_think_start=true, which is critical for
    //      correct `</think>` boundary detection in the streaming path.
    //
    // The OpenAI path handles this in the preprocessor: it renders the template,
    // inspects the formatted prompt for a trailing `<think>`, and sets
    // prompt_injected_reasoning accordingly. The Anthropic path bypasses the
    // preprocessor, so we infer prompt injection from the reasoning parser config.
    let prompt_injected_reasoning =
        parsing_options.reasoning_parser.is_some() && !thinking_explicitly_disabled;

    if prompt_injected_reasoning {
        let args = chat_request
            .chat_template_args
            .get_or_insert_with(Default::default);
        args.entry("enable_thinking".to_string())
            .or_insert(serde_json::Value::Bool(true));
        // Preserve reasoning from prior turns. Some templates (Nemotron)
        // strip historical <think> content by default to save context.
        // For agentic flows the model needs to see why it made prior decisions.
        // Ref: NVIDIA's SWE training config also sets this to false:
        // https://github.com/NVIDIA-NeMo/Nemotron/blob/main/src/nemotron/recipes/super3/stage2_rl/stage2_swe2/config/default.yaml#L287
        args.entry("truncate_history_thinking".to_string())
            .or_insert(serde_json::Value::Bool(false));
    }

    let request = context.map(|_req| chat_request);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    // Create inflight_guard early to ensure all errors are counted
    let mut inflight_guard = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::AnthropicMessages,
        streaming,
        request.id(),
    );

    tracing::trace!("Issuing generate call for Anthropic messages");

    let engine_stream = engine.generate(request).await.map_err(|e| {
        if super::metrics::request_was_rejected(e.as_ref()) {
            state
                .metrics_clone()
                .inc_rejection(&model, super::metrics::Endpoint::AnthropicMessages);
        }
        // Check for cancelled request (client disconnected before response was sent)
        if super::metrics::request_was_cancelled(e.as_ref()) {
            inflight_guard.mark_error(super::metrics::ErrorType::Cancelled);
            return anthropic_error(
                StatusCode::from_u16(499).unwrap(),
                "request_cancelled",
                &format!("Request cancelled: {}", e),
            );
        }
        inflight_guard.mark_error(super::metrics::ErrorType::Internal);
        anthropic_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "api_error",
            &format!("Failed to generate completions: {}", e),
        )
    })?;

    let ctx = engine_stream.context();

    // NOTE: We intentionally do NOT apply a reasoning parser here.
    //
    // For ModelInput::Tokens backends (skip_tokenizer_init=True), the engine
    // pipeline includes the OpenAI preprocessor which already applies reasoning
    // parsing in its backward edge (postprocessor_parsing_stream). The stream
    // arriving here already has reasoning_content and content correctly split.
    // Applying a second parser would re-classify post-think content chunks
    // (where reasoning_content=None, content=Some) as reasoning, because the
    // </think> boundary was consumed by the first parser and doesn't appear
    // in the detokenized text.
    //
    // For ModelInput::Text backends (PushRouter, no preprocessor), reasoning
    // parsing is NOT handled in the streaming path — the backend puts raw text
    // (including <think> tags) in delta.content with reasoning_content=None.
    // This is a known gap that affects all streaming handlers (OpenAI, Anthropic,
    // Responses API) equally.
    let engine_stream: Pin<
        Box<dyn futures::Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>> + Send>,
    > = Box::pin(engine_stream);

    if streaming {
        stream_handle.arm();

        use std::sync::atomic::{AtomicBool, Ordering};

        let mut converter = match anthropic_ctx {
            Some(ctx) => {
                AnthropicStreamConverter::with_context(model_for_resp, estimated_input_tokens, ctx)
            }
            None => AnthropicStreamConverter::new(model_for_resp, estimated_input_tokens),
        };
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

        let response = chat_completion_to_anthropic_response(
            chat_response,
            &model_for_resp,
            anthropic_ctx.as_ref(),
        );

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
// Model listing / retrieval (content-negotiating)
// ---------------------------------------------------------------------------

/// Build a lookup of model display_name -> context_length from model cards.
fn build_model_context_map(state: &service_v2::State) -> std::collections::HashMap<String, u32> {
    state
        .manager()
        .get_model_cards()
        .iter()
        .map(|c| (c.display_name.clone(), c.context_length))
        .collect()
}

/// Read optional env var overrides for context window and max output tokens.
fn model_env_overrides() -> (Option<u64>, Option<u64>) {
    let context_window = match std::env::var("DYN_CONTEXT_WINDOW") {
        Ok(v) => match v.parse::<u64>() {
            Ok(val) => Some(val),
            Err(_) => {
                tracing::warn!("Invalid DYN_CONTEXT_WINDOW value '{}', ignoring", v);
                None
            }
        },
        Err(_) => None,
    };
    let max_output_tokens = match std::env::var("DYN_MAX_OUTPUT_TOKENS") {
        Ok(v) => match v.parse::<u64>() {
            Ok(val) => Some(val),
            Err(_) => {
                tracing::warn!("Invalid DYN_MAX_OUTPUT_TOKENS value '{}', ignoring", v);
                None
            }
        },
        Err(_) => None,
    };
    (context_window, max_output_tokens)
}

/// Resolve context_window for a model: env override takes precedence over MDC.
fn resolve_context_window(
    model_name: &str,
    card_map: &std::collections::HashMap<String, u32>,
    env_override: Option<u64>,
) -> Option<u64> {
    env_override.or_else(|| card_map.get(model_name).map(|&cl| cl as u64))
}

/// List all models. Returns Anthropic format when `anthropic-version` header
/// is present, otherwise OpenAI format.
async fn list_models(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
) -> Result<Response, super::openai::ErrorResponse> {
    super::openai::check_ready(&state)?;

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let models: HashSet<String> = state.manager().model_display_names();
    let card_map = build_model_context_map(&state);
    let (cw_override, mot_override) = model_env_overrides();

    if headers.contains_key("anthropic-version") {
        let created_at = chrono::DateTime::from_timestamp(created as i64, 0)
            .unwrap_or_default()
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();
        let data: Vec<serde_json::Value> = models
            .iter()
            .map(|name| {
                let mut obj = serde_json::json!({
                    "id": name,
                    "display_name": name,
                    "type": "model",
                    "created_at": created_at,
                });
                if let Some(cw) = resolve_context_window(name, &card_map, cw_override) {
                    obj["max_input_tokens"] = serde_json::json!(cw);
                }
                if let Some(mot) = mot_override {
                    obj["max_tokens"] = serde_json::json!(mot);
                }
                obj
            })
            .collect();
        let first_id = data
            .first()
            .and_then(|d| d["id"].as_str().map(String::from));
        let last_id = data.last().and_then(|d| d["id"].as_str().map(String::from));
        return Ok(Json(serde_json::json!({
            "data": data,
            "has_more": false,
            "first_id": first_id,
            "last_id": last_id,
        }))
        .into_response());
    }

    // OpenAI format fallback
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|name| {
            let mut obj = serde_json::json!({
                "id": name,
                "object": "model",
                "created": created,
                "owned_by": "nvidia",
            });
            if let Some(cw) = resolve_context_window(name, &card_map, cw_override) {
                obj["context_window"] = serde_json::json!(cw);
            }
            if let Some(mot) = mot_override {
                obj["max_output_tokens"] = serde_json::json!(mot);
            }
            obj
        })
        .collect();
    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
    }))
    .into_response())
}

/// Retrieve a single model by ID. Returns Anthropic format when
/// `anthropic-version` header is present, otherwise OpenAI format.
///
/// The model ID may contain slashes (e.g. `Qwen/Qwen3.5-35B-A3B-FP8`),
/// which is why this uses a wildcard `/{*model_id}` path parameter.
async fn get_model(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Response, super::openai::ErrorResponse> {
    super::openai::check_ready(&state)?;

    // Strip leading slash from wildcard capture (axum `/{*key}` includes it).
    let model_id = model_id.strip_prefix('/').unwrap_or(&model_id);

    let models: HashSet<String> = state.manager().model_display_names();
    if !models.contains(model_id) {
        return Err(super::openai::ErrorMessage::model_not_found());
    }

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let card_map = build_model_context_map(&state);
    let (cw_override, mot_override) = model_env_overrides();
    let context_window = resolve_context_window(model_id, &card_map, cw_override);

    if headers.contains_key("anthropic-version") {
        let created_at = chrono::DateTime::from_timestamp(created as i64, 0)
            .unwrap_or_default()
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();
        let mut obj = serde_json::json!({
            "id": model_id,
            "display_name": model_id,
            "type": "model",
            "created_at": created_at,
        });
        if let Some(cw) = context_window {
            obj["max_input_tokens"] = serde_json::json!(cw);
        }
        if let Some(mot) = mot_override {
            obj["max_tokens"] = serde_json::json!(mot);
        }
        Ok(Json(obj).into_response())
    } else {
        let mut obj = serde_json::json!({
            "id": model_id,
            "object": "model",
            "created": created,
            "owned_by": "nvidia",
        });
        if let Some(cw) = context_window {
            obj["context_window"] = serde_json::json!(cw);
        }
        if let Some(mot) = mot_override {
            obj["max_output_tokens"] = serde_json::json!(mot);
        }
        Ok(Json(obj).into_response())
    }
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

/// Estimate input token count for an Anthropic request.
///
/// Uses the same heuristic as `AnthropicCountTokensRequest::estimate_tokens()`
/// (sum character lengths / 3). This populates `input_tokens` in the streaming
/// `message_start` event, since the engine only reports prompt token counts on
/// the final chunk.
fn estimate_input_tokens(req: &AnthropicCreateMessageRequest) -> u32 {
    // Build a temporary count-tokens request to reuse the existing estimator.
    let count_req = AnthropicCountTokensRequest {
        model: req.model.clone(),
        messages: req.messages.clone(),
        system: req.system.clone(),
        tools: req.tools.clone(),
    };
    count_req.estimate_tokens()
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
