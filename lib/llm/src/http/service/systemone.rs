// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::{State, rejection::JsonRejection},
    http::{HeaderMap, HeaderValue, StatusCode, header::HeaderName},
    response::{IntoResponse, Response},
    routing::post,
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{Context, ManyOut},
};
use futures::{StreamExt, stream::FuturesUnordered};
use indexmap::IndexMap;
use rand::Rng;
use serde::Serialize;
use tokio::sync::OwnedSemaphorePermit;

use super::{
    RouteDoc,
    disconnect::create_connection_monitor,
    metrics::{CancellationLabels, Endpoint, ErrorType, Metrics, SystemOnePhase},
    service_v2,
};
use crate::{
    discovery::{SystemOneExecutionSelection, SystemOneSelectionError},
    local_model::runtime_config::SGLANG_GENERATE_CAPABILITY,
    protocols::{
        Annotated,
        common::{
            llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest, timing::RequestTracker,
        },
        sglang::{generate::SglangGenerateRequest, stream::SglangGenerateStream},
        systemone::{
            SystemOneAnswer, SystemOneQuestion, SystemOneRequest, SystemOneResponse,
            SystemOneUsage, answer_from_logprobs, build_native_score_request,
            parse_candidate_scores, render_question_prompt,
        },
    },
};

pub(super) const DEFAULT_PATH: &str = "/v1/systemone";
const BODY_LIMIT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorMessage,
}

#[derive(Debug, Serialize)]
struct ErrorMessage {
    message: String,
}

struct PreparedBranch {
    index: usize,
    question_id: String,
    question: SystemOneQuestion,
    prompt_ids: Vec<u32>,
    label_ids: Vec<u32>,
    native: Option<SglangGenerateRequest>,
}

struct StartedBranch {
    prepared: PreparedBranch,
    tracker: Arc<RequestTracker>,
    stream: ManyOut<Annotated<LLMEngineOutput>>,
}

struct BranchResult {
    index: usize,
    question_id: String,
    answer: SystemOneAnswer,
    input_tokens: usize,
    cached_tokens: usize,
    placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Placement {
    worker_id: u64,
    dp_rank: u32,
}

struct DispatchError {
    status: StatusCode,
    public_message: &'static str,
    detail: String,
    retry_after: bool,
}

impl DispatchError {
    fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            public_message: "request was cancelled",
            detail: "parent request context was cancelled".to_string(),
            retry_after: false,
        }
    }

    fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            public_message: "SGLang scoring worker is unavailable",
            detail: detail.into(),
            retry_after: false,
        }
    }

    fn malformed(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            public_message: "malformed SGLang scoring response",
            detail: detail.into(),
            retry_after: false,
        }
    }

    fn backend(cause: &(dyn std::error::Error + 'static)) -> Self {
        if super::metrics::request_was_cancelled(cause) {
            return Self::cancelled();
        }
        if super::metrics::request_was_rejected(cause) {
            return Self {
                status: super::error::overload_status_code(),
                public_message: "SGLang scoring worker capacity is exhausted",
                detail: "backend capacity exhausted".to_string(),
                retry_after: true,
            };
        }
        if let Some(error) = super::error::find_canonical_error_in_chain(cause)
            && let super::error::ClientErrorAction::Respond {
                status,
                public_message,
            } = super::error::http_action_for_error(error)
        {
            return Self {
                status,
                public_message,
                detail: error.reason().as_str().to_string(),
                retry_after: error.class().normalized()
                    == dynamo_runtime::error::ErrorClass::CapacityExhausted,
            };
        }
        Self::unavailable("native scoring engine failed")
    }
}

pub fn router(state: Arc<service_v2::State>, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    (
        vec![RouteDoc::new(axum::http::Method::POST, &path).with_documentation_path(DEFAULT_PATH)],
        Router::new()
            .route(&path, post(handler))
            .layer(axum::extract::DefaultBodyLimit::max(BODY_LIMIT_BYTES))
            .with_state(state),
    )
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(ErrorBody {
            error: ErrorMessage {
                message: message.into(),
            },
        }),
    )
        .into_response()
}

fn resolve_dp_rank(reported: Option<u32>, start_rank: u32, size: u32) -> Option<u32> {
    reported.or_else(|| (size == 1).then_some(start_rank))
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

async fn spawn_blocking_with_permit<T, E, F>(
    permit: OwnedSemaphorePermit,
    task: F,
) -> Result<Result<(T, OwnedSemaphorePermit), E>, tokio::task::JoinError>
where
    T: Send + 'static,
    E: Send + 'static,
    F: FnOnce() -> Result<T, E> + Send + 'static,
{
    tokio::task::spawn_blocking(move || task().map(|output| (output, permit))).await
}

fn add_input_tokens(total: usize, next: usize, limit: usize) -> Result<usize, String> {
    let expanded = total
        .checked_add(next)
        .ok_or_else(|| "request expands beyond the System One input-token limit".to_string())?;
    if expanded > limit {
        return Err(format!(
            "request expands to {expanded} prompt tokens; the System One limit is {limit}"
        ));
    }
    Ok(expanded)
}

fn pin_to_placement(preprocessed: &mut PreprocessedRequest, placement: Placement) {
    let routing = preprocessed.routing.get_or_insert_default();
    routing.backend_instance_id = Some(placement.worker_id);
    routing.dp_rank = Some(placement.dp_rank);
}

async fn handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    request: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Response {
    let requested_model = request
        .as_ref()
        .map(|Json(request)| request.model.as_str())
        .unwrap_or("");
    let metric_model = state
        .manager()
        .metric_model_for(requested_model)
        .to_string();
    let request_id = super::openai::get_or_create_request_id(&headers);
    let mut guard = state.metrics_clone().create_inflight_guard(
        &metric_model,
        Endpoint::SystemOne,
        false,
        &request_id,
    );
    guard.mark_error(ErrorType::Cancelled);
    let response = handle_request(state.clone(), request_id, request).await;
    match response.status().as_u16() {
        status
            if status == super::error::overload_status_code().as_u16()
                && response
                    .headers()
                    .contains_key(axum::http::header::RETRY_AFTER) =>
        {
            state
                .metrics_clone()
                .inc_rejection(&metric_model, Endpoint::SystemOne);
            guard.mark_error(ErrorType::Overload);
        }
        200..=299 => guard.mark_ok(),
        400 | 413 | 415 | 422 => guard.mark_error(ErrorType::Validation),
        404 => guard.mark_error(ErrorType::NotFound),
        499 => guard.mark_error(ErrorType::Cancelled),
        503 => guard.mark_error(ErrorType::Unavailable),
        _ => guard.mark_error(ErrorType::Internal),
    }
    response
}

async fn handle_request(
    state: Arc<service_v2::State>,
    request_id: String,
    request: Result<Json<SystemOneRequest>, JsonRejection>,
) -> Response {
    let request = match request {
        Ok(Json(request)) => request,
        Err(rejection) => return error(rejection.status(), rejection.body_text()),
    };
    if let Err(validation) = request.validate() {
        return error(StatusCode::UNPROCESSABLE_ENTITY, validation.to_string());
    }
    if let Err(response) = super::openai::check_ready(&state) {
        return response.into_response();
    }

    let selection = match state
        .manager()
        .get_systemone_execution_selection(&request.model, SGLANG_GENERATE_CAPABILITY)
    {
        Ok(selection) => selection,
        Err(SystemOneSelectionError::NotFound) => {
            return error(StatusCode::NOT_FOUND, "model is not registered");
        }
        Err(SystemOneSelectionError::Unavailable) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "model has no eligible aggregate SGLang worker",
            );
        }
        Err(SystemOneSelectionError::Unsupported) => {
            return error(
                StatusCode::BAD_REQUEST,
                "model does not support aggregate SGLang scoring",
            );
        }
    };

    let branch_count = u32::try_from(request.questions.len()).unwrap_or(u32::MAX);
    let admission_permit = match state
        .systemone_admission()
        .try_acquire_many_owned(branch_count)
    {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = error(
                super::error::overload_status_code(),
                "System One request capacity is exhausted",
            );
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                HeaderValue::from_static("1"),
            );
            return response;
        }
    };

    let max_input_tokens = state.systemone_max_input_tokens();
    let preflight_started = Instant::now();
    let selection_for_preflight = selection.clone();
    let (branches, admission_permit) =
        match spawn_blocking_with_permit(admission_permit, move || {
            prepare_branches(request, &selection_for_preflight, max_input_tokens)
        })
        .await
        {
            Ok(Ok(prepared)) => prepared,
            Ok(Err((status, message))) => return error(status, message),
            Err(cause) => {
                tracing::error!(error = %cause, "System One preflight task failed");
                return error(StatusCode::INTERNAL_SERVER_ERROR, "internal server error");
            }
        };

    let metrics = state.metrics_clone();
    let metric_model = state.manager().metric_model_for(&selection.canonical_model);
    metrics.observe_systemone_phase(
        metric_model,
        SystemOnePhase::Preflight,
        preflight_started.elapsed().as_secs_f64(),
    );
    metrics.observe_systemone_work(
        metric_model,
        branches.len(),
        branches.iter().map(|branch| branch.prompt_ids.len()).sum(),
        branches.iter().map(|branch| branch.label_ids.len()).sum(),
    );

    let parent = Context::with_id_and_metadata((), request_id.clone(), Default::default());
    let parent_context = parent.context();
    let cancellation_labels = CancellationLabels {
        model: state
            .manager()
            .metric_model_for(&selection.canonical_model)
            .to_string(),
        endpoint: Endpoint::SystemOne.to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        parent_context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;
    let response = match tokio::spawn(dispatch(
        selection,
        branches,
        request_id.clone(),
        parent_context,
        admission_permit,
        metrics,
    ))
    .await
    {
        Ok(response) => response,
        Err(cause) => {
            tracing::error!(%request_id, error = %cause, "System One dispatch task failed");
            error(StatusCode::INTERNAL_SERVER_ERROR, "internal server error")
        }
    };
    connection_handle.disarm();
    response
}

fn prepare_branches(
    request: SystemOneRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
    let context_length = selection.card.effective_context_length() as usize;
    let mut total_input_tokens = 0_usize;
    let mut branches = Vec::with_capacity(request.questions.len());
    for (index, (question_id, question)) in request.questions.into_iter().enumerate() {
        let rendered = render_question_prompt(&request.state, &question)
            .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;
        let remaining_tokens = max_input_tokens.saturating_sub(total_input_tokens);
        let max_prompt_tokens = if context_length == 0 {
            remaining_tokens
        } else {
            remaining_tokens.min(context_length.saturating_sub(1))
        };
        let (prompt_ids, label_ids) = selection
            .preprocessor
            .render_systemone_question(
                &selection.canonical_model,
                &rendered.content,
                request.chat_template_kwargs.as_ref(),
                &rendered.labels,
                max_prompt_tokens,
            )
            .map_err(|cause| {
                if let Some(oversized) =
                    cause.downcast_ref::<crate::preprocessor::SystemOnePromptTooLong>()
                {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("question {question_id:?}: {oversized}"),
                    );
                }
                tracing::warn!("System One prompt rendering failed");
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "question {question_id:?}: tokenizer or chat template could not render this request"
                    ),
                )
            })?;
        total_input_tokens =
            add_input_tokens(total_input_tokens, prompt_ids.len(), max_input_tokens)
                .map_err(|message| (StatusCode::UNPROCESSABLE_ENTITY, message))?;
        let native = build_native_score_request(&prompt_ids, &label_ids, "validation")
            .and_then(|value| {
                serde_json::from_value::<SglangGenerateRequest>(value).map_err(|cause| {
                    crate::protocols::systemone::SystemOneError::CandidateScores(cause.to_string())
                })
            })
            .map_err(|cause| {
                tracing::error!(error = %cause, "failed to build System One native request");
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal server error".to_string(),
                )
            })?;
        branches.push(PreparedBranch {
            index,
            question_id,
            question,
            prompt_ids,
            label_ids,
            native: Some(native),
        });
    }
    Ok(branches)
}

async fn dispatch(
    selection: SystemOneExecutionSelection,
    mut branches: Vec<PreparedBranch>,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    _admission_permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
) -> Response {
    let mut salt_bytes = [0_u8; 16];
    rand::rng().fill(&mut salt_bytes);
    let cache_salt = URL_SAFE_NO_PAD.encode(salt_bytes);
    let first = branches.remove(0);
    let first_started = Instant::now();
    let first_result = match run_branch(
        selection.clone(),
        first,
        request_id.clone(),
        cache_salt.clone(),
        None,
        parent_context.clone(),
        metrics.clone(),
    )
    .await
    {
        Ok(result) => result,
        Err(cause) => return dispatch_error(&request_id, &parent_context, cause),
    };
    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::FirstBranch,
        first_started.elapsed().as_secs_f64(),
    );
    let placement = first_result.placement;
    let mut results = Vec::with_capacity(branches.len() + 1);
    results.push(first_result);

    let mut siblings = FuturesUnordered::new();
    let fanout_started = Instant::now();
    for branch in branches {
        siblings.push(run_branch(
            selection.clone(),
            branch,
            request_id.clone(),
            cache_salt.clone(),
            Some(placement),
            parent_context.clone(),
            metrics.clone(),
        ));
    }
    while let Some(result) = siblings.next().await {
        match result {
            Ok(result) if result.placement == placement => results.push(result),
            Ok(result) => {
                return dispatch_error(
                    &request_id,
                    &parent_context,
                    DispatchError::unavailable(format!(
                        "branch {} was routed to {:?}, expected {:?}",
                        result.index, result.placement, placement
                    )),
                );
            }
            Err(cause) => return dispatch_error(&request_id, &parent_context, cause),
        }
    }

    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::Fanout,
        fanout_started.elapsed().as_secs_f64(),
    );

    results.sort_by_key(|result| result.index);
    let mut answers = IndexMap::with_capacity(results.len());
    let mut input_tokens = 0_usize;
    for result in results {
        input_tokens = input_tokens.saturating_add(result.input_tokens);
        answers.insert(result.question_id, result.answer);
    }
    let mut response = Json(SystemOneResponse {
        model: selection.canonical_model,
        answers,
        usage: SystemOneUsage {
            input_tokens,
            output_tokens: 0,
        },
    })
    .into_response();
    response.headers_mut().insert(
        HeaderName::from_static("x-request-id"),
        HeaderValue::from_str(&request_id).unwrap_or_else(|_| HeaderValue::from_static("invalid")),
    );
    response.headers_mut().insert(
        HeaderName::from_static("x-dynamo-systemone-version"),
        HeaderValue::from_static("1"),
    );
    response
}

fn dispatch_error(
    request_id: &str,
    parent_context: &Arc<dyn AsyncEngineContext>,
    cause: DispatchError,
) -> Response {
    parent_context.kill();
    if cause.status.as_u16() == 499 {
        tracing::debug!(%request_id, "System One request cancelled");
    } else {
        tracing::error!(%request_id, error = %cause.detail, "System One scoring failed");
    }
    let mut response = error(cause.status, cause.public_message);
    if cause.retry_after {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            HeaderValue::from_static("1"),
        );
    }
    response
}

async fn run_branch(
    selection: SystemOneExecutionSelection,
    mut prepared: PreparedBranch,
    request_id: String,
    cache_salt: String,
    pinned: Option<Placement>,
    parent_context: Arc<dyn AsyncEngineContext>,
    metrics: Arc<Metrics>,
) -> Result<BranchResult, DispatchError> {
    let mut native = prepared
        .native
        .take()
        .ok_or_else(|| DispatchError::malformed("native request was already consumed"))?;
    native.cache_salt = Some(cache_salt);
    let branch_id = format!("{request_id}-{}", prepared.index);
    let mut preprocessed = super::sglang_generate::preprocessed_request(
        native,
        &selection.canonical_model,
        None,
        &branch_id,
    )
    .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))?;
    let tracker = Arc::new(RequestTracker::new());
    tracker.record_isl(prepared.prompt_ids.len(), None);
    if let Some(placement) = pinned {
        pin_to_placement(&mut preprocessed, placement);
    }
    preprocessed.tracker = Some(tracker.clone());
    let context: Context<PreprocessedRequest> =
        Context::with_id_and_metadata(preprocessed, branch_id, Default::default());
    let branch_context = context.context();
    parent_context.link_child(branch_context.clone());
    if parent_context.is_killed() {
        branch_context.kill();
        return Err(DispatchError::cancelled());
    }
    let stream = run_until_killed(parent_context.as_ref(), selection.engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| DispatchError::backend(cause.as_ref()))?;
    let stream_context = stream.context();
    parent_context.link_child(stream_context.clone());
    if parent_context.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    let result = finish_branch(
        StartedBranch {
            prepared,
            tracker,
            stream,
        },
        selection.card.runtime_config.data_parallel_start_rank,
        selection.card.runtime_config.data_parallel_size,
        parent_context,
    )
    .await?;
    metrics.observe_systemone_branch(&selection.canonical_model, result.cached_tokens);
    Ok(result)
}

async fn finish_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let StartedBranch {
        prepared,
        tracker,
        stream,
    } = started;
    let responses = SglangGenerateStream::from_annotated_stream(stream);
    tokio::pin!(responses);
    let mut terminal = None;
    loop {
        let frame = run_until_killed(parent_context.as_ref(), responses.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(frame) = frame else { break };
        let frame = frame.map_err(|cause| {
            if super::metrics::request_was_rejected(&cause)
                || super::error::find_canonical_error_in_chain(&cause).is_some()
            {
                DispatchError::backend(&cause)
            } else {
                DispatchError::malformed("invalid native scoring stream")
            }
        })?;
        if frame
            .get("meta_info")
            .and_then(|meta| meta.get("finish_reason"))
            .is_some_and(|finish| !finish.is_null())
            && terminal.replace(frame).is_some()
        {
            return Err(DispatchError::malformed(
                "multiple terminal SGLang scoring frames",
            ));
        }
    }
    let terminal = terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal SGLang scoring frame"))?;
    let placement = tracker
        .get_worker_info()
        .and_then(|worker| {
            worker
                .decode_worker_id
                .or(worker.prefill_worker_id)
                .map(|worker_id| {
                    resolve_dp_rank(
                        worker.decode_dp_rank.or(worker.prefill_dp_rank),
                        data_parallel_start_rank,
                        data_parallel_size,
                    )
                    .map(|dp_rank| Placement { worker_id, dp_rank })
                })
        })
        .flatten()
        .ok_or_else(|| {
            DispatchError::unavailable("Dynamo did not report an unambiguous scoring worker rank")
        })?;
    let scores = parse_candidate_scores(&terminal, &prepared.label_ids)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let answer = answer_from_logprobs(&prepared.question, &scores)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let input_tokens = prepared.prompt_ids.len();
    let cached_tokens = terminal
        .get("meta_info")
        .and_then(|meta| meta.get("cached_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .and_then(|tokens| usize::try_from(tokens).ok())
        .unwrap_or(0);
    Ok(BranchResult {
        index: prepared.index,
        question_id: prepared.question_id,
        answer,
        input_tokens,
        cached_tokens,
        placement,
    })
}

#[cfg(test)]
#[path = "systemone/tests.rs"]
mod tests;
