// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP boundary and response assembly for engine-native token generation.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use axum::body::{Body, to_bytes};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response, sse::Event, sse::KeepAlive, sse::Sse};
use axum::{Json, Router, middleware, routing::post};
use dynamo_protocols::types::CompletionUsage;
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider};
use futures::StreamExt;
use serde::Serialize;
use tracing::Instrument;

use super::disconnect::{create_connection_monitor, monitor_for_disconnects};
use super::error::HttpError;
use super::metrics::{CancellationLabels, Endpoint, ErrorType};
use super::openai::{
    ErrorMessage, ErrorResponse, check_model_serving_ready, check_ready, context_from_headers,
    get_body_limit, get_or_create_request_id, smart_json_error_middleware,
};
use super::{RouteDoc, service_v2};
use crate::discovery::ModelManagerError;
use crate::protocols::common::FinishReason;
use crate::protocols::common::extensions::HEADER_DATA_PARALLEL_RANK_ALIAS;
use crate::protocols::common::llm_backend::{LLMEngineOutput, TopLogprob};
use crate::protocols::inference::generate::{
    GENERATE_DP_RANK_CONTEXT_KEY, GENERATE_PATH, GenerateChoiceLogprobs, GenerateLogprob,
    GenerateProtocolError, GenerateRequest, GenerateResponse, GenerateResponseChoice,
    GenerateStreamResponse, GenerateStreamResponseChoice, GenerateTokenLogprob, GenerateTopLogprob,
};

pub(super) fn router(state: Arc<service_v2::State>) -> (Vec<RouteDoc>, Router) {
    let doc = RouteDoc::new(axum::http::Method::POST, GENERATE_PATH);
    let router = Router::new()
        .route(GENERATE_PATH, post(handler))
        .layer(middleware::from_fn(smart_json_error_middleware))
        .layer(axum::extract::DefaultBodyLimit::max(get_body_limit()))
        .layer(middleware::from_fn(vllm_error_envelope_middleware))
        .with_state(state);
    (vec![doc], router)
}

async fn handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    Json(request): Json<GenerateRequest>,
) -> Response {
    handle_generate(state, headers, request)
        .await
        .unwrap_or_else(vllm_error_response)
}

async fn handle_generate(
    state: Arc<service_v2::State>,
    headers: HeaderMap,
    mut request: GenerateRequest,
) -> Result<Response, ErrorResponse> {
    check_ready(&state)?;
    request.validate().map_err(protocol_error)?;

    let model = resolve_model(&state, request.model.as_deref())?;
    check_model_serving_ready(&state, &model)?;
    request.model = Some(model.clone());

    let request_id = resolve_request_id(&headers, request.request_id.as_deref());
    request.request_id = Some(request_id.clone());
    let require_logprobs = request.sampling_params.logprobs.is_some();
    let expected_choices = request.sampling_params.n.unwrap_or(1);
    let streaming = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);

    let cancellation_labels = CancellationLabels {
        model: state.manager().metric_model_for(&model).to_string(),
        endpoint: Endpoint::InferenceGenerate.to_string(),
        request_type: if streaming { "stream" } else { "unary" }.to_string(),
    };
    let mut request = context_from_headers(request, request_id.clone(), &headers)?;
    if let Some(dp_rank) = data_parallel_rank_from_headers(&headers) {
        request.insert(GENERATE_DP_RANK_CONTEXT_KEY, dp_rank);
    }
    let context = request.context();
    let (mut connection_handle, stream_handle) =
        create_connection_monitor(context, Some(state.metrics_clone()), cancellation_labels).await;

    if streaming {
        let response = generate_streaming(
            state,
            request,
            request_id,
            model,
            expected_choices,
            require_logprobs,
            include_usage,
            continuous_usage,
            stream_handle,
        )
        .await;
        connection_handle.disarm();
        return response;
    }

    let response = match tokio::spawn(
        generate_unary(
            state,
            request,
            request_id,
            model,
            expected_choices,
            require_logprobs,
            stream_handle,
        )
        .in_current_span(),
    )
    .await
    {
        Ok(response) => response,
        Err(error) => Err(ErrorMessage::internal_server_error_with_details(
            "Failed to await token generation task",
            format!("{error:?}"),
        )),
    };
    connection_handle.disarm();
    let response = response?;
    Ok(Json(response).into_response())
}

#[derive(Serialize)]
struct VllmErrorEnvelope {
    error: VllmErrorBody,
}

#[derive(Serialize)]
struct VllmErrorBody {
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
    code: u16,
}

fn vllm_error_type(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "invalid_request_error",
        404 => "not_found",
        499 => "request_cancelled",
        500 => "internal_error",
        501 => "not_implemented",
        503 => "service_unavailable",
        429 | 529 => "overloaded",
        _ if status.is_client_error() => "invalid_request_error",
        _ => "internal_error",
    }
}

fn vllm_error_body(status: StatusCode, message: String) -> Response {
    (
        status,
        Json(VllmErrorEnvelope {
            error: VllmErrorBody {
                message,
                error_type: vllm_error_type(status),
                code: status.as_u16(),
            },
        }),
    )
        .into_response()
}

fn vllm_error_response((status, Json(error)): ErrorResponse) -> Response {
    let message = serde_json::to_value(error)
        .ok()
        .and_then(|value| {
            value
                .get("message")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| "Internal server error".to_string());
    vllm_error_body(status, message)
}

async fn vllm_error_envelope_middleware(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    if !response.status().is_client_error() && !response.status().is_server_error() {
        return response;
    }

    let status = response.status();
    let (mut parts, body) = response.into_parts();
    let bytes = match to_bytes(body, get_body_limit()).await {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::error!(%error, "failed to read generate error response body");
            return vllm_error_body(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal server error".to_string(),
            );
        }
    };
    if serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .is_some_and(|value| value.get("error").is_some())
    {
        parts.headers.remove(header::CONTENT_LENGTH);
        return Response::from_parts(parts, Body::from(bytes));
    }
    let message = if status == StatusCode::INTERNAL_SERVER_ERROR {
        "Internal server error".to_string()
    } else {
        serde_json::from_slice::<serde_json::Value>(&bytes)
            .ok()
            .and_then(|value| {
                value
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned())
    };
    vllm_error_body(status, message)
}

#[allow(clippy::too_many_arguments)]
async fn generate_streaming(
    state: Arc<service_v2::State>,
    request: dynamo_runtime::pipeline::Context<GenerateRequest>,
    request_id: String,
    model: String,
    expected_choices: u32,
    require_logprobs: bool,
    include_usage: bool,
    continuous_usage: bool,
    stream_handle: super::disconnect::ConnectionHandle,
) -> Result<axum::response::Response, ErrorResponse> {
    let mut inflight = state.metrics_clone().create_inflight_guard(
        &model,
        Endpoint::InferenceGenerate,
        true,
        &request_id,
    );
    let engine = state
        .manager()
        .get_generate_engine(&model)
        .map_err(|error| {
            let response = ErrorMessage::from_model_error(&error);
            inflight.mark_error(match error {
                ModelManagerError::ModelUnavailable(_) => ErrorType::Unavailable,
                _ => ErrorType::NotFound,
            });
            response
        })?;
    let stream = engine.generate(request).await.map_err(|error| {
        let response = ErrorMessage::from_anyhow(error, "Failed to generate tokens");
        inflight.mark_error(ErrorType::Internal);
        response
    })?;
    let context = stream.context();
    let events = generate_event_stream(
        stream,
        GenerateStreamAccumulator::new(request_id, expected_choices as usize, require_logprobs),
        include_usage,
        continuous_usage,
    );
    let events = monitor_for_disconnects(events, context, inflight, stream_handle);
    let mut response = Sse::new(events);
    if let Some(keep_alive) = state.sse_keep_alive() {
        response = response.keep_alive(KeepAlive::default().interval(keep_alive));
    }
    Ok(response.into_response())
}

fn generate_event_stream(
    stream: impl futures::Stream<Item = crate::types::Annotated<LLMEngineOutput>> + Send + 'static,
    mut accumulator: GenerateStreamAccumulator,
    include_usage: bool,
    continuous_usage: bool,
) -> impl futures::Stream<Item = Result<Event, axum::Error>> {
    async_stream::stream! {
        tokio::pin!(stream);
        let mut failed = false;
        while let Some(mut annotated) = stream.next().await {
            let result = if let Some(error) = annotated.error.take() {
                Err(invalid_response(format!("backend generation failed: {error}")))
            } else if annotated.event.as_deref() == Some("error") {
                Err(invalid_response(
                    annotated
                        .comment
                        .as_ref()
                        .map(|parts| parts.join(", "))
                        .unwrap_or_else(|| "backend generation failed".to_string()),
                ))
            } else if let Some(output) = annotated.data.take() {
                accumulator.push(output, continuous_usage)
            } else {
                Ok(None)
            };

            match result.and_then(serialize_stream_chunk) {
                Ok(Some(event)) => yield Ok(event),
                Ok(None) => {}
                Err(error) => {
                    failed = true;
                    yield Err(stream_error(error));
                    break;
                }
            }
        }

        if !failed {
            match accumulator.finish(include_usage).and_then(serialize_stream_chunk) {
                Ok(Some(event)) => yield Ok(event),
                Ok(None) => {}
                Err(error) => yield Err(stream_error(error)),
            }
        }
    }
}

fn serialize_stream_chunk(
    response: Option<GenerateStreamResponse>,
) -> Result<Option<Event>, GenerateProtocolError> {
    response
        .map(|response| {
            serde_json::to_string(&response)
                .map(|json| Event::default().data(json))
                .map_err(|error| {
                    invalid_response(format!("failed to encode stream chunk: {error}"))
                })
        })
        .transpose()
}

fn stream_error(error: GenerateProtocolError) -> axum::Error {
    axum::Error::new(std::io::Error::other(error.to_string()))
}

async fn run_until_killed<T>(
    context: &dyn AsyncEngineContext,
    operation: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::pin!(operation);
    tokio::select! {
        biased;

        // Keep an ownership-bearing result when completion races with a kill.
        // Callers re-check both contexts before consuming the result.
        result = &mut operation => Some(result),
        _ = context.killed() => None,
    }
}

fn cancelled_error() -> ErrorResponse {
    ErrorMessage::from_http_error(HttpError {
        code: 499,
        message: "request was cancelled".to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
async fn generate_unary(
    state: Arc<service_v2::State>,
    request: dynamo_runtime::pipeline::Context<GenerateRequest>,
    request_id: String,
    model: String,
    expected_choices: u32,
    require_logprobs: bool,
    _stream_handle: super::disconnect::ConnectionHandle,
) -> Result<GenerateResponse, ErrorResponse> {
    let request_context = request.context();
    let mut inflight = state.metrics_clone().create_inflight_guard(
        state.manager().metric_model_for(&model),
        Endpoint::InferenceGenerate,
        false,
        &request_id,
    );
    let engine = state
        .manager()
        .get_generate_engine(&model)
        .map_err(|error| {
            let response = ErrorMessage::from_model_error(&error);
            inflight.mark_error(match error {
                ModelManagerError::ModelUnavailable(_) => ErrorType::Unavailable,
                _ => ErrorType::NotFound,
            });
            response
        })?;
    let generate_result =
        match run_until_killed(request_context.as_ref(), engine.generate(request)).await {
            Some(result) => result,
            None => {
                inflight.mark_error(ErrorType::Cancelled);
                return Err(cancelled_error());
            }
        };
    if request_context.is_killed() {
        inflight.mark_error(ErrorType::Cancelled);
        return Err(cancelled_error());
    }
    let mut stream = match generate_result {
        Ok(stream) => stream,
        Err(error) => {
            let was_cancelled = request_context.is_killed()
                || super::metrics::request_was_cancelled(error.as_ref());
            inflight.mark_error(if was_cancelled {
                ErrorType::Cancelled
            } else {
                ErrorType::Internal
            });
            if was_cancelled {
                return Err(cancelled_error());
            }
            return Err(ErrorMessage::from_anyhow(
                error,
                "Failed to generate tokens",
            ));
        }
    };
    let engine_context = stream.context();
    let mut accumulator =
        GenerateAccumulator::new(request_id, expected_choices as usize, require_logprobs);

    loop {
        let next = match run_until_killed(request_context.as_ref(), stream.next()).await {
            Some(next) => next,
            None => {
                inflight.mark_error(ErrorType::Cancelled);
                return Err(cancelled_error());
            }
        };
        if request_context.is_killed() || engine_context.is_killed() {
            inflight.mark_error(ErrorType::Cancelled);
            return Err(cancelled_error());
        }
        let Some(mut event) = next else {
            break;
        };
        if let Some(error) = event.error.take() {
            let was_cancelled = super::metrics::request_was_cancelled(&error);
            inflight.mark_error(if was_cancelled {
                ErrorType::Cancelled
            } else {
                ErrorType::Internal
            });
            if was_cancelled {
                return Err(cancelled_error());
            }
            return Err(ErrorMessage::from_anyhow(
                anyhow::Error::new(error),
                "Token generation failed",
            ));
        }
        if event.event.as_deref() == Some("error") {
            let details = event
                .comment
                .as_ref()
                .map(|parts| parts.join(", "))
                .unwrap_or_else(|| "backend error".to_string());
            inflight.mark_error(ErrorType::Internal);
            return Err(ErrorMessage::internal_server_error_with_details(
                "Token generation failed",
                details,
            ));
        }
        if let Some(output) = event.data.take() {
            if matches!(output.finish_reason.as_ref(), Some(FinishReason::Cancelled)) {
                inflight.mark_error(ErrorType::Cancelled);
                return Err(cancelled_error());
            }
            accumulator.push(output).map_err(|error| {
                inflight.mark_error(ErrorType::Internal);
                protocol_error(error)
            })?;
        }
    }

    if request_context.is_killed() || engine_context.is_killed() {
        inflight.mark_error(ErrorType::Cancelled);
        return Err(cancelled_error());
    }
    let response = accumulator.finish().map_err(|error| {
        inflight.mark_error(ErrorType::Internal);
        protocol_error(error)
    })?;
    inflight.mark_ok();
    Ok(response)
}

fn resolve_request_id(headers: &HeaderMap, body_request_id: Option<&str>) -> String {
    headers
        .get("x-request-id")
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or_else(|| body_request_id.map(ToOwned::to_owned))
        .unwrap_or_else(|| get_or_create_request_id(headers))
}

fn data_parallel_rank_from_headers(headers: &HeaderMap) -> Option<u32> {
    headers
        .get(HEADER_DATA_PARALLEL_RANK_ALIAS)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse().ok())
}

struct GenerateStreamAccumulator {
    request_id: String,
    expected_choices: usize,
    require_logprobs: bool,
    seen_choices: BTreeSet<u32>,
    terminal_choices: BTreeSet<u32>,
    completion_tokens: BTreeMap<u32, usize>,
    latest_usage: Option<CompletionUsage>,
}

impl GenerateStreamAccumulator {
    fn new(request_id: String, expected_choices: usize, require_logprobs: bool) -> Self {
        Self {
            request_id,
            expected_choices,
            require_logprobs,
            seen_choices: BTreeSet::new(),
            terminal_choices: BTreeSet::new(),
            completion_tokens: BTreeMap::new(),
            latest_usage: None,
        }
    }

    fn push(
        &mut self,
        output: LLMEngineOutput,
        continuous_usage: bool,
    ) -> Result<Option<GenerateStreamResponse>, GenerateProtocolError> {
        let index = output.index.unwrap_or(0);
        if index as usize >= self.expected_choices {
            return Err(invalid_response(format!(
                "choice {index} exceeds requested cardinality {}",
                self.expected_choices
            )));
        }
        if self.terminal_choices.contains(&index) {
            return Err(invalid_response(format!(
                "choice {index} emitted data after its terminal chunk"
            )));
        }
        self.seen_choices.insert(index);

        let token_ids = output.token_ids;
        let logprobs = if self.require_logprobs && !token_ids.is_empty() {
            let selected = output.log_probs.as_ref().ok_or_else(|| {
                invalid_response(format!(
                    "choice {index} omitted requested completion logprobs"
                ))
            })?;
            let candidates = output.top_logprobs.as_ref().ok_or_else(|| {
                invalid_response(format!("choice {index} omitted requested top logprobs"))
            })?;
            if selected.len() != token_ids.len() || candidates.len() != token_ids.len() {
                return Err(invalid_response(format!(
                    "choice {index} token/logprob counts are not aligned"
                )));
            }
            Some(GenerateChoiceLogprobs {
                content: Some(
                    token_ids
                        .iter()
                        .zip(selected)
                        .zip(candidates)
                        .map(|((token_id, logprob), top)| {
                            build_token_logprob(*token_id, *logprob, top)
                        })
                        .collect(),
                ),
                refusal: None,
            })
        } else {
            None
        };

        *self.completion_tokens.entry(index).or_default() += token_ids.len();
        if let Some(usage) = output.completion_usage {
            self.latest_usage = Some(usage);
        }
        let finish_reason = output.finish_reason.map(public_finish_reason).transpose()?;
        if finish_reason.is_some() {
            self.terminal_choices.insert(index);
        }

        if token_ids.is_empty() && finish_reason.is_none() {
            return Ok(None);
        }
        let usage = continuous_usage
            .then(|| self.normalized_usage())
            .transpose()?
            .flatten();
        Ok(Some(GenerateStreamResponse {
            request_id: self.request_id.clone(),
            choices: vec![GenerateStreamResponseChoice {
                index,
                logprobs,
                finish_reason,
                token_ids,
            }],
            usage,
        }))
    }

    fn finish(
        self,
        include_usage: bool,
    ) -> Result<Option<GenerateStreamResponse>, GenerateProtocolError> {
        if self.seen_choices.len() != self.expected_choices
            || self.terminal_choices.len() != self.expected_choices
        {
            return Err(invalid_response(format!(
                "expected {} terminal choices, saw {} choices and {} terminals",
                self.expected_choices,
                self.seen_choices.len(),
                self.terminal_choices.len()
            )));
        }
        if !include_usage {
            return Ok(None);
        }
        let usage = self.normalized_usage()?.ok_or_else(|| {
            invalid_response("backend omitted usage requested for the final stream chunk")
        })?;
        Ok(Some(GenerateStreamResponse {
            request_id: self.request_id,
            choices: Vec::new(),
            usage: Some(usage),
        }))
    }

    fn normalized_usage(&self) -> Result<Option<CompletionUsage>, GenerateProtocolError> {
        let Some(mut usage) = self.latest_usage.clone() else {
            return Ok(None);
        };
        let completion_tokens = self.completion_tokens.values().sum::<usize>();
        usage.completion_tokens = u32::try_from(completion_tokens)
            .map_err(|_| invalid_response("completion token count exceeds u32"))?;
        usage.total_tokens = usage.prompt_tokens.saturating_add(usage.completion_tokens);
        Ok(Some(usage))
    }
}

fn resolve_model(
    state: &service_v2::State,
    requested: Option<&str>,
) -> Result<String, ErrorResponse> {
    if let Some(model) = requested {
        return Ok(model.to_string());
    }
    let mut models = state.manager().list_generate_models();
    models.sort();
    match models.as_slice() {
        [model] => Ok(model.clone()),
        [] => Err(ErrorMessage::model_not_found()),
        _ => Err(ErrorMessage::from_http_error(HttpError {
            code: 400,
            message: "Validation: `model` is required when multiple models are served".into(),
        })),
    }
}

struct GenerateAccumulator {
    request_id: String,
    choices: BTreeMap<u32, ChoiceAccumulator>,
    prompt_logprobs: Option<
        Vec<
            Option<
                std::collections::HashMap<
                    u32,
                    crate::protocols::inference::generate::GenerateLogprob,
                >,
            >,
        >,
    >,
    kv_transfer_params: Option<serde_json::Value>,
    expected_choices: usize,
    require_logprobs: bool,
}

#[derive(Default)]
struct ChoiceAccumulator {
    token_ids: Vec<u32>,
    logprobs: Vec<GenerateTokenLogprob>,
    finish_reason: Option<String>,
    routed_experts: Option<String>,
}

impl GenerateAccumulator {
    fn new(request_id: String, expected_choices: usize, require_logprobs: bool) -> Self {
        Self {
            request_id,
            choices: BTreeMap::new(),
            prompt_logprobs: None,
            kv_transfer_params: None,
            expected_choices,
            require_logprobs,
        }
    }

    fn push(&mut self, output: LLMEngineOutput) -> Result<(), GenerateProtocolError> {
        let index = output.index.unwrap_or(0);
        if index as usize >= self.expected_choices {
            return Err(invalid_response(format!(
                "choice {index} exceeds requested cardinality {}",
                self.expected_choices
            )));
        }
        let choice = self.choices.entry(index).or_default();
        if choice.finish_reason.is_some() {
            return Err(invalid_response(format!(
                "choice {index} emitted data after its terminal chunk"
            )));
        }

        if self.require_logprobs && !output.token_ids.is_empty() {
            let selected = output.log_probs.as_ref().ok_or_else(|| {
                invalid_response(format!(
                    "choice {index} omitted requested completion logprobs"
                ))
            })?;
            let candidates = output.top_logprobs.as_ref().ok_or_else(|| {
                invalid_response(format!("choice {index} omitted requested top logprobs"))
            })?;
            if selected.len() != output.token_ids.len()
                || candidates.len() != output.token_ids.len()
            {
                return Err(invalid_response(format!(
                    "choice {index} token/logprob counts are not aligned"
                )));
            }
            for ((token_id, logprob), top) in output.token_ids.iter().zip(selected).zip(candidates)
            {
                choice
                    .logprobs
                    .push(build_token_logprob(*token_id, *logprob, top));
            }
        }
        choice.token_ids.extend(output.token_ids);

        if let Some(metadata) = output.generate_metadata {
            if let Some(mut prompt_logprobs) = metadata.prompt_logprobs {
                normalize_prompt_logprobs(&mut prompt_logprobs);
                if self
                    .prompt_logprobs
                    .as_ref()
                    .is_some_and(|current| current != &prompt_logprobs)
                {
                    return Err(invalid_response(
                        "workers returned inconsistent prompt logprobs",
                    ));
                }
                self.prompt_logprobs = Some(prompt_logprobs);
            }
            if let Some(kv_transfer_params) = metadata.kv_transfer_params {
                self.kv_transfer_params = Some(kv_transfer_params);
            }
            if let Some(routed_experts) = metadata.routed_experts {
                choice.routed_experts = Some(routed_experts);
            }
        }
        if let Some(reason) = output.finish_reason {
            choice.finish_reason = Some(public_finish_reason(reason)?);
        }
        Ok(())
    }

    fn finish(self) -> Result<GenerateResponse, GenerateProtocolError> {
        if self.choices.len() != self.expected_choices {
            return Err(invalid_response(format!(
                "expected {} choices, received {}",
                self.expected_choices,
                self.choices.len()
            )));
        }
        let choices = self
            .choices
            .into_iter()
            .map(|(index, choice)| {
                let finish_reason = choice.finish_reason.ok_or_else(|| {
                    invalid_response(format!("choice {index} has no terminal finish reason"))
                })?;
                if self.require_logprobs && choice.logprobs.len() != choice.token_ids.len() {
                    return Err(invalid_response(format!(
                        "choice {index} accumulated token/logprob counts are not aligned"
                    )));
                }
                Ok(GenerateResponseChoice {
                    index,
                    logprobs: self.require_logprobs.then_some(GenerateChoiceLogprobs {
                        content: Some(choice.logprobs),
                        refusal: None,
                    }),
                    finish_reason: Some(finish_reason),
                    token_ids: choice.token_ids,
                    routed_experts: choice.routed_experts,
                })
            })
            .collect::<Result<Vec<_>, GenerateProtocolError>>()?;
        Ok(GenerateResponse {
            request_id: self.request_id,
            choices,
            prompt_logprobs: self.prompt_logprobs,
            kv_transfer_params: self.kv_transfer_params,
        })
    }
}

fn clamp_vllm_logprob(logprob: f64) -> f32 {
    let logprob = logprob as f32;
    if logprob.is_finite() {
        logprob.max(-9999.0)
    } else {
        -9999.0
    }
}

fn token_bytes(token: &str, backend_bytes: Option<&Vec<u8>>) -> Option<Vec<u8>> {
    backend_bytes
        .cloned()
        .or_else(|| (!token.is_empty()).then(|| token.as_bytes().to_vec()))
}

fn build_token_logprob(token_id: u32, selected: f64, top: &[TopLogprob]) -> GenerateTokenLogprob {
    let selected_candidate = top.iter().find(|candidate| candidate.token_id == token_id);
    let token = selected_candidate
        .and_then(|candidate| candidate.token.clone())
        .unwrap_or_else(|| format!("token_id:{token_id}"));
    let bytes = token_bytes(
        &token,
        selected_candidate.and_then(|candidate| candidate.bytes.as_ref()),
    );
    let selected = clamp_vllm_logprob(selected);
    let mut top_logprobs = top
        .iter()
        .map(|candidate| {
            let token = candidate
                .token
                .clone()
                .unwrap_or_else(|| format!("token_id:{}", candidate.token_id));
            GenerateTopLogprob {
                bytes: token_bytes(&token, candidate.bytes.as_ref()),
                token,
                logprob: clamp_vllm_logprob(candidate.logprob),
            }
        })
        .collect::<Vec<_>>();
    if selected_candidate.is_none() {
        top_logprobs.push(GenerateTopLogprob {
            token: token.clone(),
            logprob: selected,
            bytes: bytes.clone(),
        });
    }
    GenerateTokenLogprob {
        token,
        logprob: selected,
        bytes,
        top_logprobs,
    }
}

fn normalize_prompt_logprobs(
    prompt_logprobs: &mut [Option<std::collections::HashMap<u32, GenerateLogprob>>],
) {
    for entries in prompt_logprobs.iter_mut().flatten() {
        for entry in entries.values_mut() {
            entry.logprob = clamp_vllm_logprob(f64::from(entry.logprob));
        }
    }
}

fn public_finish_reason(reason: FinishReason) -> Result<String, GenerateProtocolError> {
    match reason {
        FinishReason::EoS | FinishReason::Stop => Ok("stop".to_string()),
        FinishReason::Length => Ok("length".to_string()),
        FinishReason::ContentFilter => Ok("content_filter".to_string()),
        FinishReason::Cancelled => Err(invalid_response("backend generation was cancelled")),
        FinishReason::Error(message) => Err(invalid_response(format!(
            "backend generation failed: {message}"
        ))),
    }
}

fn protocol_error(error: GenerateProtocolError) -> ErrorResponse {
    match error {
        GenerateProtocolError::InvalidRequest(message) => {
            ErrorMessage::from_http_error(HttpError {
                code: 400,
                message: format!("Validation: {message}"),
            })
        }
        GenerateProtocolError::InvalidResponse(message) => {
            ErrorMessage::internal_server_error_with_details(
                "Token generation returned an invalid response",
                message,
            )
        }
    }
}

fn invalid_response(message: impl Into<String>) -> GenerateProtocolError {
    GenerateProtocolError::InvalidResponse(message.into())
}

#[cfg(test)]
mod tests;
