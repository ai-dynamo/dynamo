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
    service_v2,
};
use crate::{
    discovery::{SystemOneEngine, SystemOneExecutionSelection, SystemOneSelectionError},
    generate_format::{Bound, ClientRequest, GenerateExchange, Worker},
    protocols::{
        Annotated,
        common::{
            FinishReason, llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest,
            timing::RequestTracker,
        },
        response_merge::PREFILL_OUTPUT_ANNOTATION,
        systemone::{
            PoolingDecisionConfig, ScoringBackend, VllmScoreCollector, build_native_score_request,
            build_pooling_request, causal_scoring_request, extract_marker_logits,
            format_chunked_question_text, format_joint_user_prompt, format_pooling_state_text,
            parse_candidate_scores, reduce_head_logits,
        },
    },
};
use dynamo_http_server::{
    disconnect::create_connection_monitor,
    metrics::{CancellationLabels, Endpoint, Metrics, SystemOnePhase},
};

#[path = "systemone/abort.rs"]
mod abort;
#[path = "systemone/backend_error.rs"]
mod backend_error;
#[path = "systemone/response.rs"]
mod response;
pub(super) use response::attach_request_ids;
use response::{decision_error, error};

#[path = "systemone/http.rs"]
mod http;
pub use http::router;

pub(super) const DEFAULT_PATH: &str = "/v1/systemone";
pub(super) const DECISIONS_PATH: &str = "/v1/decisions";

// The configured input ceiling and deployment candidate ceiling bound amplification.
const MAX_TOKENIZATION_WORK: usize = 1_048_576 * (255 + 2);
const MAX_TOKENIZATION_TEXT_BYTES: usize = 16 * 1024 * 1024;
const MAX_CONCURRENT_SIBLINGS: usize = 4;

#[derive(Clone, Copy)]
pub(super) struct DecisionCapacityRejection;

fn metric_endpoint(dialect: Dialect) -> Endpoint {
    match dialect {
        Dialect::Jev => Endpoint::SystemOne,
        Dialect::OpenAi => Endpoint::Decisions,
    }
}

struct PreparedBranch {
    index: usize,
    temperature: f64,
    prompt_ids: Vec<u32>,
    label_ids: Vec<u32>,
    native: Option<serde_json::Value>,
}

struct PreparedPoolingBatch {
    question_indices: Vec<usize>,
    token_ids: Vec<u32>,
    markers: Vec<usize>,
}

enum PreparedExecution {
    Causal(Vec<PreparedBranch>),
    Pooling(Vec<PreparedPoolingBatch>),
}

impl PreparedExecution {
    fn work_summary(&self) -> (usize, usize, usize) {
        match self {
            Self::Causal(branches) => (
                branches.len(),
                branches.iter().map(|b| b.prompt_ids.len()).sum(),
                branches.iter().map(|b| b.label_ids.len()).sum(),
            ),
            Self::Pooling(batches) => (
                batches.len(),
                batches.iter().map(|b| b.token_ids.len()).sum(),
                batches.iter().map(|b| b.markers.len()).sum(),
            ),
        }
    }
}

struct StartedBranch {
    backend: ScoringBackend,
    prepared: PreparedBranch,
    tracker: Arc<RequestTracker>,
    stream: ManyOut<Annotated<LLMEngineOutput>>,
    exchange: Box<dyn GenerateExchange>,
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
    is_capacity_exhausted: bool,
}

impl DispatchError {
    fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            public_message: "request was cancelled",
            detail: "parent request context was cancelled".to_string(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            public_message: "scoring worker is unavailable",
            detail: detail.into(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn malformed(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            public_message: "malformed scoring response",
            detail: detail.into(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn worker_failed(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            public_message: "scoring worker failed",
            detail: detail.into(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
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
            decision_lifecycle::AdmissionError::InvalidBranchCount => error(
                dialect,
                StatusCode::UNPROCESSABLE_ENTITY,
                "decision requests require at least one question",
            ),
            decision_lifecycle::AdmissionError::AdmissionClosed => error(
                dialect,
                StatusCode::SERVICE_UNAVAILABLE,
                "decision admission is closed",
            ),
            decision_lifecycle::AdmissionError::RequestTooLarge => error(
                dialect,
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "request needs {branches} branches; the System One branch limit is {limit}"
                ),
            ),
            decision_lifecycle::AdmissionError::CapacityExhausted => response::capacity_error(
                dialect,
                dynamo_error::http::overload_status_code(),
                "System One request capacity is exhausted",
            ),
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

fn tokenization_work_limit(input_tokens: usize, candidates: usize) -> Option<usize> {
    input_tokens
        .checked_mul(candidates.checked_add(2)?)
        .filter(|work| *work <= MAX_TOKENIZATION_WORK)
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
    if request.questions.is_empty() {
        return decision_error(DecisionError::validation(
            dialect,
            "request must contain at least one question",
        ));
    }
    let deadline = tokio::time::Instant::now() + state.systemone_request_timeout();
    if super::openai::check_ready(&state).is_err() {
        return error(
            dialect,
            StatusCode::SERVICE_UNAVAILABLE,
            "service is not ready",
        );
    }

    let required_candidates = request
        .questions
        .iter()
        .map(|question| question.candidates.len())
        .max()
        .unwrap_or_default();
    let selection = match state
        .manager()
        .get_systemone_execution_selection(&request.model, required_candidates)
    {
        Ok(selection) => selection,
        Err(SystemOneSelectionError::NotFound) => {
            return error(dialect, StatusCode::NOT_FOUND, "model is not registered");
        }
        Err(SystemOneSelectionError::Unavailable) => {
            return error(
                dialect,
                StatusCode::SERVICE_UNAVAILABLE,
                "model has no eligible scoring worker",
            );
        }
        Err(SystemOneSelectionError::Unsupported) => {
            return error(
                dialect,
                StatusCode::BAD_REQUEST,
                "model does not support decision scoring",
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
            return response::capacity_error(
                dialect,
                dynamo_error::http::overload_status_code(),
                "System One preprocessing capacity is exhausted",
            );
        }
    };
    let preflight_started = Instant::now();
    let selection_for_preflight = selection.clone();
    let request = Arc::new(request);
    let preflight_request = request.clone();
    let (prepared, admission_permit) = match tokio::time::timeout_at(
        deadline,
        spawn_blocking_with_permit(admission_permit, move || {
            let _preflight_permit = preflight_permit;
            match selection_for_preflight.backend {
                ScoringBackend::VllmPooling => prepare_pooling_batches(
                    &preflight_request,
                    &selection_for_preflight,
                    max_input_tokens,
                )
                .map(PreparedExecution::Pooling),
                ScoringBackend::Sglang | ScoringBackend::Vllm => prepare_branches(
                    &preflight_request,
                    &selection_for_preflight,
                    max_input_tokens,
                )
                .map(PreparedExecution::Causal),
            }
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
    let (work_batches, work_prompt_tokens, work_labels) = prepared.work_summary();
    metrics.observe_systemone_work(metric_model, work_batches, work_prompt_tokens, work_labels);

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
    let task = match prepared {
        PreparedExecution::Causal(branches) => tokio::spawn(dispatch(
            selection,
            branches,
            request_id.clone(),
            parent_context.clone(),
            admission_permit,
            metrics,
            request,
        )),
        PreparedExecution::Pooling(batches) => tokio::spawn(dispatch_pooling(
            selection,
            batches,
            request_id.clone(),
            parent_context.clone(),
            admission_permit,
            metrics,
            request,
        )),
    };
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
    let mut remaining_tokenization_work = tokenization_work_limit(
        max_input_tokens,
        selection.decision_capabilities().max_candidates,
    )
    .ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "System One tokenization work configuration exceeds its supported limit".to_string(),
        )
    })?;
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
                if let Some(incompatible) = cause
                    .downcast_ref::<crate::preprocessor::SystemOneCandidateLabelsIncompatible>()
                {
                    return (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        format!("question {question_id:?}: {incompatible}"),
                    );
                }
                let reason = if let Some(incompatible) =
                    cause.downcast_ref::<crate::preprocessor::SystemOneModelIncompatible>()
                {
                    incompatible.0
                } else if dynamo_error::http::find_canonical_error_in_chain(cause.as_ref())
                    .is_some_and(|error| {
                        error.class() == dynamo_runtime::error::ErrorClass::InvalidRequest
                    })
                {
                    "chat template or tokenizer failed"
                } else {
                    tracing::error!("System One prompt tokenization failed");
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "internal server error".to_string(),
                    );
                };
                tracing::warn!(reason, "System One prompt rendering failed");
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
        let native = match selection.backend {
            ScoringBackend::Sglang => {
                build_native_score_request(&prompt_ids, &label_ids, "validation")
            }
            ScoringBackend::Vllm => causal_scoring_request(&prompt_ids, &label_ids, "validation"),
            ScoringBackend::VllmPooling => {
                unreachable!("pooling decisions use prepare_pooling_batches")
            }
        }
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
    // Share sibling prefixes, but isolate requests until a trusted tenant scope exists.
    // Client metadata and request IDs are not authenticated cache identities.
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
        output_tokens: (results.len() * selection.backend.output_tokens()) as u64,
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
    finish_decision_response(
        &request,
        &selection.canonical_model,
        &outcomes,
        &usage,
        &request_id,
    )
}

fn finish_decision_response(
    request: &CanonicalRequest,
    model: &str,
    outcomes: &[QuestionOutcome],
    usage: &Usage,
    request_id: &str,
) -> Response {
    let mut response = match project_response(request, model, outcomes, usage) {
        Ok(body) => Json(body).into_response(),
        Err(cause) => decision_error(cause),
    };
    attach_request_ids(&mut response, request_id);
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
    if cause.is_capacity_exhausted {
        return response::capacity_error(dialect, cause.status, cause.public_message);
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
    native["cache_salt"] = serde_json::Value::String(cache_salt);
    let branch_id = format!("{request_id}-{}", prepared.index);
    let Bound {
        request: mut preprocessed,
        exchange,
    } = bind_native(
        selection.backend,
        &selection.canonical_model,
        selection.card.kv_cache_block_size,
        &native,
        &branch_id,
    )
    .map_err(|cause| DispatchError::malformed(format!("native request: {cause:#}")))?;
    preprocessed
        .routing
        .get_or_insert_default()
        .expected_output_tokens = Some(selection.backend.output_tokens() as u32);
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
    let generate_engine = match &selection.engine {
        SystemOneEngine::Generate(engine) => engine,
        SystemOneEngine::Pooling(_) => {
            return Err(DispatchError::malformed(
                "pooling engine cannot execute causal scoring branch",
            ));
        }
    };
    let stream = run_until_killed(parent_context.as_ref(), generate_engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    let stream_context = stream.context();
    parent_context.link_child(stream_context.clone());
    if parent_context.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    let result = finish_branch(
        StartedBranch {
            backend: selection.backend,
            prepared,
            tracker,
            stream,
            exchange,
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

/// Bind the backend's native request to the existing Generate transport.
fn bind_native(
    backend: ScoringBackend,
    model: &str,
    kv_cache_block_size: u32,
    native: &serde_json::Value,
    request_id: &str,
) -> anyhow::Result<Bound> {
    let format = crate::generate_format::named(backend.format())?;
    let body = serde_json::to_vec(native)?;
    let mut headers = axum::http::HeaderMap::new();
    headers.insert(
        axum::http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    let opened = format
        .open(ClientRequest {
            body: &body,
            headers: &headers,
            default_request_id: request_id,
        })
        .map_err(|error| anyhow::anyhow!(error.message))?;
    opened
        .validation
        .map_err(|error| anyhow::anyhow!(error.message))?;
    (opened.bind)(&Worker {
        model,
        kv_cache_block_size,
        lora_name: None,
        tower_connector_lora_enabled: false,
    })
    .map_err(|error| anyhow::anyhow!(error.message))
}

fn stream_error(cause: dynamo_runtime::error::DynamoError) -> DispatchError {
    if cause.class() == dynamo_runtime::error::ErrorClass::BackendProtocol {
        DispatchError::malformed("invalid native scoring stream")
    } else {
        backend_error::classify(&cause)
    }
}

fn accept_frame(
    frame: serde_json::Value,
    terminal: &mut Option<serde_json::Value>,
) -> Result<(), DispatchError> {
    let finish = frame
        .get("meta_info")
        .and_then(|meta| meta.get("finish_reason"));
    if let Some(cause) = finish.and_then(abort::classify) {
        return Err(cause);
    }
    if finish.is_some_and(|finish| !finish.is_null()) && terminal.replace(frame).is_some() {
        return Err(DispatchError::malformed(
            "multiple terminal SGLang scoring frames",
        ));
    }
    Ok(())
}

async fn finish_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let result = match started.backend {
        ScoringBackend::Sglang => {
            read_sglang_branch(
                started,
                data_parallel_start_rank,
                data_parallel_size,
                parent_context.clone(),
            )
            .await
        }
        ScoringBackend::Vllm => {
            let StartedBranch {
                prepared,
                tracker,
                stream,
                exchange,
                ..
            } = started;
            drop(exchange);
            read_vllm_branch(
                prepared,
                tracker,
                stream,
                data_parallel_start_rank,
                data_parallel_size,
                parent_context.clone(),
            )
            .await
        }
        ScoringBackend::VllmPooling => {
            unreachable!("pooling decisions use dispatch_pooling")
        }
    };
    if parent_context.is_killed() {
        return Err(DispatchError::cancelled());
    }
    result
}

async fn read_sglang_branch(
    started: StartedBranch,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let StartedBranch {
        prepared,
        tracker,
        mut stream,
        mut exchange,
        ..
    } = started;
    let mut terminal = None;
    let mut frames = Vec::new();
    loop {
        let delta = run_until_killed(parent_context.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(delta) = delta else { break };
        let prefill = delta.event.as_deref() == Some(PREFILL_OUTPUT_ANNOTATION);
        if let Some(output) = delta.into_data().map_err(stream_error)? {
            exchange
                .chunk(output, prefill, &mut |frame| frames.push(frame.clone()))
                .map_err(stream_error)?;
        }
        for frame in frames.drain(..) {
            accept_frame(frame, &mut terminal)?;
        }
    }
    if let Some(frame) = exchange.finish().map_err(stream_error)? {
        accept_frame(frame, &mut terminal)?;
    }
    let terminal = terminal
        .ok_or_else(|| DispatchError::malformed("missing terminal SGLang scoring frame"))?;
    let placement = branch_placement(&tracker, data_parallel_start_rank, data_parallel_size)?;
    let scores = parse_candidate_scores(&terminal, &prepared.label_ids)
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens: prepared.prompt_ids.len(),
        cached_tokens: measured_cached_tokens(&terminal, prepared.prompt_ids.len())?,
        placement,
    })
}

async fn read_vllm_branch(
    prepared: PreparedBranch,
    tracker: Arc<RequestTracker>,
    mut stream: ManyOut<Annotated<LLMEngineOutput>>,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
    parent_context: Arc<dyn AsyncEngineContext>,
) -> Result<BranchResult, DispatchError> {
    let mut collector = VllmScoreCollector::new(prepared.label_ids, prepared.prompt_ids.len())
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    loop {
        let delta = run_until_killed(parent_context.as_ref(), stream.next())
            .await
            .ok_or_else(DispatchError::cancelled)?;
        let Some(delta) = delta else { break };
        if let Some(output) = delta.into_data().map_err(stream_error)? {
            match &output.finish_reason {
                Some(FinishReason::Cancelled) => {
                    return Err(DispatchError::unavailable(
                        "native scoring worker cancelled",
                    ));
                }
                Some(FinishReason::Error(detail)) => {
                    return Err(DispatchError::worker_failed(detail.clone()));
                }
                _ => {}
            }
            collector
                .observe(&output)
                .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        }
    }
    let result = collector
        .finish()
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
    let answer = QuestionOutcome::Answer(
        reduce_vocab_logprobs(&result.scores, prepared.temperature)
            .map_err(|cause| DispatchError::malformed(cause.to_string()))?,
    );
    Ok(BranchResult {
        index: prepared.index,
        answer,
        input_tokens: prepared.prompt_ids.len(),
        cached_tokens: Some(result.cached_tokens),
        placement: branch_placement(&tracker, data_parallel_start_rank, data_parallel_size)?,
    })
}

fn branch_placement(
    tracker: &RequestTracker,
    data_parallel_start_rank: u32,
    data_parallel_size: u32,
) -> Result<Placement, DispatchError> {
    tracker
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

struct PoolingPrep<'a> {
    request: &'a CanonicalRequest,
    selection: &'a SystemOneExecutionSelection,
    config: &'a PoolingDecisionConfig,
    marker_id: u32,
    max_seq_len: usize,
    max_per_batch: usize,
    remaining_encoding_bytes: usize,
    batches: Vec<PreparedPoolingBatch>,
}

impl<'a> PoolingPrep<'a> {
    fn prepare_chunked(
        &mut self,
        questions: &[&dynamo_decisions::CanonicalQuestion],
    ) -> Result<(), (StatusCode, String)> {
        let separator = self.config.separator.as_deref().ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "chunked pooling decision config missing separator".to_string(),
            )
        })?;
        let separator_id = self
            .selection
            .preprocessor
            .systemone_single_token_id(
                separator,
                "pooling decision separator must encode to a single token",
            )
            .map_err(map_pooling_render_error)?;
        let state_text = format_pooling_state_text(
            &self.request.input,
            self.config,
            self.selection.card.special_tokens(),
        );
        let (context_ids, ctx_bytes) = self
            .selection
            .preprocessor
            .render_systemone_chunked_pooling_context(
                &self.selection.canonical_model,
                &state_text,
                Some(&self.request.chat_template_kwargs),
                separator_id,
                self.remaining_encoding_bytes,
            )
            .map_err(map_pooling_render_error)?;
        self.remaining_encoding_bytes = self.remaining_encoding_bytes.saturating_sub(ctx_bytes);

        self.batches.reserve(questions.len());
        for question in questions {
            let question_text = format_chunked_question_text(
                question,
                self.request.dialect,
                self.config,
                self.selection.card.special_tokens(),
            )
            .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;
            let (q_ids, q_bytes) = self
                .selection
                .preprocessor
                .encode_systemone_chunked_pooling_question(
                    &question_text,
                    self.marker_id,
                    self.remaining_encoding_bytes,
                )
                .map_err(map_pooling_render_error)?;
            self.remaining_encoding_bytes = self.remaining_encoding_bytes.saturating_sub(q_bytes);

            let mut token_ids = Vec::with_capacity(context_ids.len() + q_ids.len());
            token_ids.extend_from_slice(&context_ids);
            token_ids.extend_from_slice(&q_ids);

            let marker_count = token_ids.iter().filter(|&&t| t == self.marker_id).count();
            if marker_count != 1 || token_ids.last() != Some(&self.marker_id) {
                return Err((
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "readout markers ({marker_count}) did not match expected question after tokenization"
                    ),
                ));
            }
            if token_ids.len() > self.max_seq_len {
                return Err(pooling_overlength_error(
                    question,
                    token_ids.len(),
                    self.max_seq_len,
                ));
            }
            let marker_pos = token_ids.len() - 1;
            self.batches.push(PreparedPoolingBatch {
                question_indices: vec![question.ordinal],
                token_ids,
                markers: vec![marker_pos],
            });
        }
        Ok(())
    }

    fn bisect(
        &mut self,
        questions: &[&dynamo_decisions::CanonicalQuestion],
    ) -> Result<(), (StatusCode, String)> {
        if questions.len() > self.max_per_batch {
            let mid = questions.len() / 2;
            self.bisect(&questions[..mid])?;
            return self.bisect(&questions[mid..]);
        }

        let user_content = format_joint_user_prompt(
            &self.request.input,
            questions,
            self.request.dialect,
            self.config,
            self.selection.card.special_tokens(),
        )
        .map_err(|cause| (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string()))?;

        let (token_ids, markers, encoding_bytes) = self
            .selection
            .preprocessor
            .render_systemone_pooling_prompt(
                &self.selection.canonical_model,
                &user_content,
                Some(&self.request.chat_template_kwargs),
                self.marker_id,
                questions.len(),
                self.remaining_encoding_bytes,
            )
            .map_err(map_pooling_render_error)?;
        self.remaining_encoding_bytes =
            self.remaining_encoding_bytes.saturating_sub(encoding_bytes);

        if token_ids.len() <= self.max_seq_len {
            self.batches.push(PreparedPoolingBatch {
                question_indices: questions.iter().map(|q| q.ordinal).collect(),
                token_ids,
                markers,
            });
            return Ok(());
        }
        if questions.len() == 1 {
            return Err(pooling_overlength_error(
                questions[0],
                token_ids.len(),
                self.max_seq_len,
            ));
        }
        let mid = questions.len() / 2;
        self.bisect(&questions[..mid])?;
        self.bisect(&questions[mid..])
    }
}

fn prepare_pooling_batches(
    request: &CanonicalRequest,
    selection: &SystemOneExecutionSelection,
    max_input_tokens: usize,
) -> Result<Vec<PreparedPoolingBatch>, (StatusCode, String)> {
    let config = selection.pooling_config.as_ref().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "missing pooling decision config for selected worker".to_string(),
        )
    })?;
    let marker_id = selection
        .preprocessor
        .systemone_single_token_id(
            &config.marker,
            "pooling decision readout marker must encode to a single token",
        )
        .map_err(map_pooling_render_error)?;
    let context_length = selection.card.effective_context_length() as usize;
    let max_seq_len = match (context_length, config.max_length) {
        (0, Some(limit)) => limit,
        (0, None) => max_input_tokens,
        (ctx, Some(limit)) => ctx.min(limit),
        (ctx, None) => ctx,
    };
    let questions: Vec<&dynamo_decisions::CanonicalQuestion> = request.questions.iter().collect();
    let mut prep = PoolingPrep {
        request,
        selection,
        config,
        marker_id,
        max_seq_len,
        max_per_batch: config.max_questions.unwrap_or(usize::MAX),
        remaining_encoding_bytes: MAX_TOKENIZATION_TEXT_BYTES,
        batches: Vec::new(),
    };
    if config.is_chunked() {
        prep.prepare_chunked(&questions)?;
    } else {
        prep.bisect(&questions)?;
    }

    let mut total_input_tokens = 0_usize;
    for batch in &prep.batches {
        total_input_tokens =
            add_input_tokens(total_input_tokens, batch.token_ids.len(), max_input_tokens)
                .map_err(|message| (StatusCode::UNPROCESSABLE_ENTITY, message))?;
    }
    Ok(prep.batches)
}

fn map_pooling_render_error(cause: anyhow::Error) -> (StatusCode, String) {
    if let Some(oversized) = cause.downcast_ref::<crate::preprocessor::SystemOneEncodingTooLarge>()
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, oversized.to_string());
    }
    if let Some(incompatible) =
        cause.downcast_ref::<crate::preprocessor::SystemOneModelIncompatible>()
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, incompatible.0.to_string());
    }
    if dynamo_error::http::find_canonical_error_in_chain(cause.as_ref())
        .is_some_and(|e| e.class() == dynamo_runtime::error::ErrorClass::InvalidRequest)
    {
        return (StatusCode::UNPROCESSABLE_ENTITY, cause.to_string());
    }
    tracing::error!(error = %cause, "pooling decision prompt rendering failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal server error".to_string(),
    )
}

fn pooling_overlength_error(
    question: &dynamo_decisions::CanonicalQuestion,
    tokens: usize,
    max_seq_len: usize,
) -> (StatusCode, String) {
    let qid = question.id.as_deref().unwrap_or("unnamed");
    (
        StatusCode::UNPROCESSABLE_ENTITY,
        format!(
            "question {qid:?} needs {tokens} tokens with the state, exceeding the {max_seq_len}-token limit; inputs are never truncated"
        ),
    )
}

fn link_child_or_cancelled(
    parent: &Arc<dyn AsyncEngineContext>,
    child: Arc<dyn AsyncEngineContext>,
) -> Result<(), DispatchError> {
    parent.link_child(child.clone());
    if parent.is_killed() {
        child.kill();
        return Err(DispatchError::cancelled());
    }
    Ok(())
}

async fn dispatch_pooling(
    selection: SystemOneExecutionSelection,
    batches: Vec<PreparedPoolingBatch>,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    _admission_permit: OwnedSemaphorePermit,
    metrics: Arc<Metrics>,
    request: Arc<CanonicalRequest>,
) -> Response {
    let dialect = request.dialect;
    let Some(config) = selection.pooling_config.clone() else {
        return dispatch_error(
            &request_id,
            &parent_context,
            DispatchError::malformed("missing pooling decision config"),
            dialect,
        );
    };
    let fanout_started = Instant::now();

    let mut futures_stream =
        stream::iter(batches.into_iter().enumerate().map(|(batch_idx, batch)| {
            run_pooling_batch(
                selection.clone(),
                config.clone(),
                batch_idx,
                batch,
                request_id.clone(),
                parent_context.clone(),
                request.clone(),
            )
        }))
        .buffer_unordered(MAX_CONCURRENT_SIBLINGS);

    let mut per_question: Vec<(usize, QuestionOutcome)> =
        Vec::with_capacity(request.questions.len());
    let mut total_input_tokens: u64 = 0;
    let mut failure = None;
    while let Some(result) = futures_stream.next().await {
        if failure.is_some() {
            continue;
        }
        match result {
            Ok((input_tokens, batch_outcomes)) => {
                total_input_tokens += input_tokens as u64;
                per_question.extend(batch_outcomes);
            }
            Err(cause) => {
                parent_context.kill();
                failure = Some(cause);
            }
        }
    }
    drop(futures_stream);
    if let Some(cause) = failure {
        return dispatch_error(&request_id, &parent_context, cause, dialect);
    }

    metrics.observe_systemone_phase(
        &selection.canonical_model,
        SystemOnePhase::Fanout,
        fanout_started.elapsed().as_secs_f64(),
    );

    per_question.sort_by_key(|(ordinal, _)| *ordinal);
    let outcomes: Vec<QuestionOutcome> = per_question.into_iter().map(|(_, o)| o).collect();
    let usage = Usage {
        input_tokens: total_input_tokens,
        output_tokens: 0,
        cached_tokens: Some(0),
        reasoning_tokens: 0,
    };
    finish_decision_response(
        &request,
        &selection.canonical_model,
        &outcomes,
        &usage,
        &request_id,
    )
}

async fn run_pooling_batch(
    selection: SystemOneExecutionSelection,
    config: PoolingDecisionConfig,
    batch_idx: usize,
    batch: PreparedPoolingBatch,
    request_id: String,
    parent_context: Arc<dyn AsyncEngineContext>,
    request: Arc<CanonicalRequest>,
) -> Result<(usize, Vec<(usize, QuestionOutcome)>), DispatchError> {
    selection.validate_dispatch(None).map_err(|_| {
        DispatchError::unavailable("selected worker revision is no longer available")
    })?;
    let SystemOneEngine::Pooling(pooling_engine) = &selection.engine else {
        return Err(DispatchError::malformed(
            "generate engine cannot execute pooling decision batch",
        ));
    };
    let seq_len = batch.token_ids.len();
    let task = if config.uses_last_classify() {
        "classify"
    } else {
        "token_classify"
    };
    let pooling_req = build_pooling_request(&selection.canonical_model, batch.token_ids, task);
    let context = Context::with_id_and_metadata(
        pooling_req,
        format!("{request_id}-p{batch_idx}"),
        Default::default(),
    );
    link_child_or_cancelled(&parent_context, context.context())?;
    let mut stream = run_until_killed(parent_context.as_ref(), pooling_engine.generate(context))
        .await
        .ok_or_else(DispatchError::cancelled)?
        .map_err(|cause| backend_error::classify(cause.as_ref()))?;
    link_child_or_cancelled(&parent_context, stream.context())?;

    let mut response = None;
    while let Some(frame) = run_until_killed(parent_context.as_ref(), stream.next())
        .await
        .ok_or_else(DispatchError::cancelled)?
    {
        if let Some(payload) = frame.into_data().map_err(stream_error)? {
            if response.replace(payload).is_some() {
                return Err(DispatchError::malformed(
                    "pooling worker returned multiple payloads",
                ));
            }
        }
    }
    if parent_context.is_killed() {
        return Err(DispatchError::cancelled());
    }
    let response =
        response.ok_or_else(|| DispatchError::malformed("pooling worker returned no payload"))?;

    let rows = extract_marker_logits(
        &response,
        seq_len,
        config.num_classes(),
        &batch.markers,
        config.readout_offset,
    )
    .map_err(|cause| DispatchError::malformed(cause.to_string()))?;

    let effective_temperature = config.temperature * request.temperature;
    let mut outcomes = Vec::with_capacity(batch.question_indices.len());
    for (&ordinal, row) in batch.question_indices.iter().zip(&rows) {
        let question = request
            .questions
            .iter()
            .find(|q| q.ordinal == ordinal)
            .ok_or_else(|| DispatchError::malformed("unknown question ordinal in pooling batch"))?;
        let distribution = reduce_head_logits(
            row,
            question.candidates.len(),
            question.kind == dynamo_decisions::QuestionKind::Predicate,
            effective_temperature,
        )
        .map_err(|cause| DispatchError::malformed(cause.to_string()))?;
        outcomes.push((ordinal, QuestionOutcome::Answer(distribution)));
    }
    Ok((seq_len, outcomes))
}

#[cfg(test)]
#[path = "systemone/tests.rs"]
mod tests;
