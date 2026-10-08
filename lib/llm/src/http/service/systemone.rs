// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json,
    http::{HeaderValue, StatusCode, header::HeaderName},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use dynamo_decisions::{
    CanonicalRequest, DecisionError, Dialect, QuestionOutcome, Usage, project_response,
    reduce_vocab_logprobs, render_question_prompt,
};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{Context, ManyOut},
};
use futures::{StreamExt, stream};
use rand::Rng;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use super::{
    decision_lifecycle::{self, run_until_killed, spawn_blocking_with_permit},
    disconnect::create_connection_monitor,
    metrics::{CancellationLabels, Endpoint, Metrics, SystemOnePhase},
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
        systemone::{build_native_score_request, parse_candidate_scores},
    },
};

#[path = "systemone/response.rs"]
mod response;
use response::{attach_request_ids, decision_error, error};

#[path = "systemone/http.rs"]
mod http;
pub use http::router;

pub(super) const DEFAULT_PATH: &str = "/v1/systemone";
pub(super) const DECISIONS_PATH: &str = "/v1/decisions";

const MAX_TOKENIZATION_WORK: usize = 1_048_576;
const MAX_TOKENIZATION_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CONCURRENT_SIBLINGS: usize = 4;

fn metric_endpoint(dialect: Dialect) -> Endpoint {
    match dialect {
        Dialect::Jev => Endpoint::SystemOne,
        Dialect::OpenAi | Dialect::SglangNative => Endpoint::Decisions,
    }
}

struct PreparedBranch {
    index: usize,
    temperature: f64,
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
    answer: QuestionOutcome,
    input_tokens: usize,
    cached_tokens: Option<usize>,
    placement: Placement,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Placement {
    worker_id: u64,
    dp_rank: u32,
}

#[derive(Debug)]
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
                status: StatusCode::TOO_MANY_REQUESTS,
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
                status: if error.class().normalized()
                    == dynamo_runtime::error::ErrorClass::CapacityExhausted
                {
                    StatusCode::TOO_MANY_REQUESTS
                } else {
                    status
                },
                public_message,
                detail: error.reason().as_str().to_string(),
                retry_after: error.class().normalized()
                    == dynamo_runtime::error::ErrorClass::CapacityExhausted,
            };
        }
        Self::unavailable("native scoring engine failed")
    }
}

fn acquire_admission(
    admission: Arc<Semaphore>,
    branches: u32,
    limit: usize,
    dialect: Dialect,
) -> Result<OwnedSemaphorePermit, Box<Response>> {
    decision_lifecycle::acquire_admission(admission, branches, limit).map_err(|cause| {
        let response = match cause {
            decision_lifecycle::AdmissionError::RequestTooLarge => error(
                dialect,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "request needs {branches} branches; the System One branch limit is {limit}"
                ),
            ),
            decision_lifecycle::AdmissionError::CapacityExhausted => {
                let mut response = error(
                    dialect,
                    StatusCode::TOO_MANY_REQUESTS,
                    "System One request capacity is exhausted",
                );
                response.headers_mut().insert(
                    axum::http::header::RETRY_AFTER,
                    HeaderValue::from_static("1"),
                );
                response
            }
        };
        Box::new(response)
    })
}

fn resolve_dp_rank(reported: Option<u32>, start_rank: u32, size: u32) -> Option<u32> {
    reported.or_else(|| (size == 1).then_some(start_rank))
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

fn encoding_prompt_limit(remaining_work: usize, candidates: usize) -> usize {
    remaining_work / (candidates + 2)
}

fn pin_to_placement(preprocessed: &mut PreprocessedRequest, placement: Placement) {
    let routing = preprocessed.routing.get_or_insert_default();
    routing.backend_instance_id = Some(placement.worker_id);
    routing.dp_rank = Some(placement.dp_rank);
}

async fn handle_request(
    state: Arc<service_v2::State>,
    request_id: String,
    request: CanonicalRequest,
) -> Response {
    let dialect = request.dialect;
    let deadline = tokio::time::Instant::now() + state.systemone_request_timeout();
    if super::openai::check_ready(&state).is_err() {
        return error(
            dialect,
            StatusCode::SERVICE_UNAVAILABLE,
            "service is not ready",
        );
    }

    let selection = match state
        .manager()
        .get_systemone_execution_selection(&request.model, SGLANG_GENERATE_CAPABILITY)
    {
        Ok(selection) => selection,
        Err(SystemOneSelectionError::NotFound) => {
            return error(dialect, StatusCode::NOT_FOUND, "model is not registered");
        }
        Err(SystemOneSelectionError::Unavailable) => {
            return error(
                dialect,
                StatusCode::SERVICE_UNAVAILABLE,
                "model has no eligible aggregate SGLang worker",
            );
        }
        Err(SystemOneSelectionError::Unsupported) => {
            return error(
                dialect,
                StatusCode::BAD_REQUEST,
                "model does not support aggregate SGLang scoring",
            );
        }
    };
    if let Err(cause) = request.validate_capabilities(&selection.decision_capabilities()) {
        return decision_error(cause);
    }

    let branch_count = u32::try_from(request.questions.len()).unwrap_or(u32::MAX);
    let admission_permit = match acquire_admission(
        state.systemone_admission(),
        branch_count,
        state.systemone_max_inflight_branches(),
        dialect,
    ) {
        Ok(permit) => permit,
        Err(response) => return *response,
    };

    let max_input_tokens = state.systemone_max_input_tokens();
    let preflight_permit = match state.systemone_preflight_admission().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = error(
                dialect,
                StatusCode::TOO_MANY_REQUESTS,
                "System One preprocessing capacity is exhausted",
            );
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                HeaderValue::from_static("1"),
            );
            return response;
        }
    };
    let preflight_started = Instant::now();
    let selection_for_preflight = selection.clone();
    let request = Arc::new(request);
    let preflight_request = request.clone();
    let (branches, admission_permit) = match tokio::time::timeout_at(
        deadline,
        spawn_blocking_with_permit(admission_permit, move || {
            let _preflight_permit = preflight_permit;
            prepare_branches(
                &preflight_request,
                &selection_for_preflight,
                max_input_tokens,
            )
        }),
    )
    .await
    {
        Ok(Ok(Ok(prepared))) => prepared,
        Ok(Ok(Err((status, message)))) => return error(dialect, status, message),
        Err(_) => {
            return error(
                dialect,
                StatusCode::GATEWAY_TIMEOUT,
                "decision request deadline exceeded",
            );
        }
        Ok(Err(cause)) => {
            tracing::error!(error = %cause, "System One preflight task failed");
            return error(
                dialect,
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal server error",
            );
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
        endpoint: metric_endpoint(dialect).to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        parent_context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;
    let task = tokio::spawn(dispatch(
        selection,
        branches,
        request_id.clone(),
        parent_context.clone(),
        admission_permit,
        metrics,
        request,
    ));
    let response = response::dispatch_with_deadline(task, deadline, parent_context, dialect).await;
    connection_handle.disarm();
    response
}

fn prepare_branches(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedBranch>, (StatusCode, String)> {
    let context_length = selection.card.effective_context_length() as usize;
    let mut total_input_tokens = 0_usize;
    let mut remaining_tokenization_work = MAX_TOKENIZATION_WORK;
    let mut remaining_encoding_bytes = MAX_TOKENIZATION_TEXT_BYTES;
    let mut branches = Vec::with_capacity(request.questions.len());
    for question in &request.questions {
        let index = question.ordinal;
        let question_id = question.id.as_deref().unwrap_or("unnamed");
        let rendered = render_question_prompt(&request.input, question)
            .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;
        let remaining_tokens = max_input_tokens.saturating_sub(total_input_tokens);
        let max_prompt_tokens = if context_length == 0 {
            remaining_tokens
        } else {
            remaining_tokens.min(context_length.saturating_sub(1))
        }
        .min(encoding_prompt_limit(
            remaining_tokenization_work,
            rendered.labels.len(),
        ));
        let (prompt_ids, label_ids, encoding_bytes) = selection
            .preprocessor
            .render_systemone_question(
                &selection.canonical_model,
                &rendered.content,
                Some(&request.chat_template_kwargs),
                &rendered.labels,
                (max_prompt_tokens, remaining_encoding_bytes),
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
                if let Some(oversized) =
                    cause.downcast_ref::<crate::preprocessor::SystemOneEncodingTooLarge>()
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
        remaining_tokenization_work -= prompt_ids.len() * (rendered.labels.len() + 2);
        remaining_encoding_bytes -= encoding_bytes;
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
            temperature: request.temperature,
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
    request: Arc<CanonicalRequest>,
) -> Response {
    let dialect = request.dialect;
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
        Err(cause) => return dispatch_error(&request_id, &parent_context, cause, dialect),
    };
    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::FirstBranch,
        first_started.elapsed().as_secs_f64(),
    );
    let placement = first_result.placement;
    let mut results = Vec::with_capacity(branches.len() + 1);
    results.push(first_result);

    let fanout_started = Instant::now();
    let mut siblings = stream::iter(branches.into_iter().map(|branch| {
        run_branch(
            selection.clone(),
            branch,
            request_id.clone(),
            cache_salt.clone(),
            Some(placement),
            parent_context.clone(),
            metrics.clone(),
        )
    }))
    .buffer_unordered(MAX_CONCURRENT_SIBLINGS);
    let mut failure = None;
    while let Some(result) = siblings.next().await {
        if failure.is_some() {
            continue;
        }
        match result {
            Ok(result) if result.placement == placement => results.push(result),
            Ok(result) => {
                parent_context.kill();
                failure = Some(DispatchError::unavailable(format!(
                    "branch {} was routed to {:?}, expected {:?}",
                    result.index, result.placement, placement
                )));
            }
            Err(cause) => {
                parent_context.kill();
                failure = Some(cause);
            }
        }
    }
    drop(siblings);
    if let Some(cause) = failure {
        return dispatch_error(&request_id, &parent_context, cause, dialect);
    }

    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::Fanout,
        fanout_started.elapsed().as_secs_f64(),
    );

    results.sort_by_key(|result| result.index);
    let mut outcomes = Vec::with_capacity(results.len());
    let mut usage = Usage {
        cached_tokens: Some(0),
        ..Default::default()
    };
    for result in results {
        usage.input_tokens += result.input_tokens as u64;
        usage.cached_tokens = usage
            .cached_tokens
            .zip(result.cached_tokens)
            .map(|(total, next)| total + next as u64);
        outcomes.push(result.answer);
    }
    let mut response =
        match project_response(&request, &selection.canonical_model, &outcomes, &usage) {
            Ok(body) => Json(body).into_response(),
            Err(cause) => decision_error(cause),
        };
    attach_request_ids(&mut response, &request_id);
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
    dialect: Dialect,
) -> Response {
    parent_context.kill();
    if cause.status.as_u16() == 499 {
        tracing::debug!(%request_id, "System One request cancelled");
    } else {
        tracing::error!(%request_id, error = %cause.detail, "System One scoring failed");
    }
    let mut response = error(dialect, cause.status, cause.public_message);
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
    selection
        .validate_dispatch(pinned.map(|placement| placement.worker_id))
        .map_err(|_| {
            DispatchError::unavailable("selected worker revision is no longer available")
        })?;
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
    if let Some(cached_tokens) = result.cached_tokens {
        metrics.observe_systemone_branch(&selection.canonical_model, cached_tokens);
    }
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
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    let input_tokens = prepared.prompt_ids.len();
    let cached_tokens = measured_cached_tokens(&terminal, input_tokens)?;
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens,
        cached_tokens,
        placement,
    })
}

fn measured_cached_tokens(
    response: &serde_json::Value,
    input_tokens: usize,
) -> Result<Option<usize>, DispatchError> {
    let Some(count) = response
        .get("meta_info")
        .and_then(|meta| meta.get("cached_tokens"))
    else {
        return Ok(None);
    };
    let count = count
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value <= input_tokens)
        .ok_or_else(|| DispatchError::malformed("invalid cache-read token count"))?;
    Ok(Some(count))
}

#[cfg(test)]
#[path = "systemone/tests.rs"]
mod tests;
