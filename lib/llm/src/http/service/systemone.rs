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
    render_question_prompt,
};
use dynamo_runtime::{
    engine::{AsyncEngineContext, AsyncEngineContextProvider},
    pipeline::{Context, ManyOut},
};
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
    generate_format::{Bound, ClientRequest, GenerateExchange, Worker},
    protocols::{
        Annotated,
        common::{
            llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest, timing::RequestTracker,
        },
        systemone::{MAX_DECISION_ENCODING_BYTES, MAX_DECISION_INPUT_TOKENS},
    },
};

#[path = "systemone/abort.rs"]
mod abort;
#[path = "systemone/backend_error.rs"]
mod backend_error;

#[path = "systemone/execution.rs"]
mod execution;
#[path = "systemone/token_execution_legacy.rs"]
mod token_execution;
use token_execution::backend::PreparedScoringRequest;
#[cfg(test)]
use token_execution::backend::measured_cached_tokens;

#[path = "systemone/response.rs"]
mod response;
pub(super) use response::attach_request_ids;
use response::{decision_error, error};

#[path = "systemone/native.rs"]
mod native;

#[path = "systemone/http.rs"]
mod http;
pub use http::router;

pub(super) const DEFAULT_PATH: &str = "/v1/systemone";
pub(super) const DECISIONS_PATH: &str = "/v1/decisions";

// The configured input ceiling and deployment candidate ceiling bound amplification.
const MAX_TOKENIZATION_WORK: usize = MAX_DECISION_INPUT_TOKENS * (255 + 2);
const MAX_TOKENIZATION_TEXT_BYTES: usize = MAX_DECISION_ENCODING_BYTES;
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
    scoring: ScoringBranch,
    wire: PreparedScoringRequest,
}

struct ScoringBranch {
    index: usize,
    temperature: f64,
    prompt_len: usize,
    label_ids: Vec<u32>,
}

struct StartedBranch {
    prepared: ScoringBranch,
    tracker: Arc<RequestTracker>,
    stream: ManyOut<Annotated<LLMEngineOutput>>,
    exchange: Box<dyn GenerateExchange>,
}

struct BranchResult {
    index: usize,
    answer: QuestionOutcome,
    input_tokens: usize,
    output_tokens: usize,
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
    code: Option<&'static str>,
    public_message: &'static str,
    detail: String,
    retry_after: bool,
    is_capacity_exhausted: bool,
}

impl DispatchError {
    fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            code: None,
            public_message: "request was cancelled",
            detail: "parent request context was cancelled".to_string(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            code: None,
            public_message: "decision scoring worker is unavailable",
            detail: detail.into(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn malformed(detail: impl Into<String>) -> Self {
        // Frontend construction/serialization failure, not executor evidence.
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: None,
            public_message: "decision request construction failed",
            detail: detail.into(),
            retry_after: false,
            is_capacity_exhausted: false,
        }
    }

    fn invalid_executor(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            code: Some("invalid_executor_result"),
            public_message: "invalid decision executor response",
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
                super::error::overload_status_code(),
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

    let selection = match state
        .manager()
        .get_systemone_execution_selection(&request.model)
    {
        Ok(selection) => selection,
        Err(SystemOneSelectionError::NotFound) => {
            return error(dialect, StatusCode::NOT_FOUND, "model is not registered");
        }
        Err(SystemOneSelectionError::Unavailable) => {
            return error(
                dialect,
                StatusCode::SERVICE_UNAVAILABLE,
                "model has no eligible aggregate decision worker",
            );
        }
        Err(SystemOneSelectionError::Unsupported) => {
            return error(
                dialect,
                StatusCode::BAD_REQUEST,
                "model does not support aggregate decision scoring",
            );
        }
    };
    if let Err(cause) = request.validate_capabilities(&selection.context.decision_capabilities()) {
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
                super::error::overload_status_code(),
                "System One preprocessing capacity is exhausted",
            );
        }
    };
    let preflight_started = Instant::now();
    let metric_model = state
        .manager()
        .metric_model_for(&selection.context.canonical_model)
        .to_owned();
    let request = Arc::new(request);
    let preflight_request = request.clone();
    let (branches, admission_permit) = match tokio::time::timeout_at(
        deadline,
        spawn_blocking_with_permit(admission_permit, move || {
            let _preflight_permit = preflight_permit;
            execution::prepare(&preflight_request, selection, max_input_tokens)
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
    metrics.observe_systemone_phase(
        &metric_model,
        SystemOnePhase::Preflight,
        preflight_started.elapsed().as_secs_f64(),
    );
    execution::observe_work(&metric_model, &branches, &metrics);

    let parent = Context::with_id_and_metadata((), request_id.clone(), Default::default());
    let parent_context = parent.context();
    let cancellation_labels = CancellationLabels {
        model: metric_model.clone(),
        endpoint: metric_endpoint(dialect).to_string(),
        request_type: "unary".to_string(),
    };
    let (mut connection_handle, _stream_handle) = create_connection_monitor(
        parent_context.clone(),
        Some(state.metrics_clone()),
        cancellation_labels,
    )
    .await;
    let task = tokio::spawn(execution::dispatch(
        branches,
        request_id.clone(),
        parent_context.clone(),
        admission_permit,
        metrics,
        metric_model,
        request,
    ));
    let response = response::dispatch_with_deadline(task, deadline, parent_context, dialect).await;
    connection_handle.disarm();
    response
}

fn fresh_cache_salt() -> String {
    let mut salt = [0_u8; 16];
    // Share sibling prefixes, but isolate requests until a trusted tenant scope exists.
    // Client metadata and request IDs are not authenticated cache identities.
    rand::rng().fill(&mut salt);
    URL_SAFE_NO_PAD.encode(salt)
}

fn finish_response(
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
    let mut response = match cause.code {
        Some(code) => response::decision_error(DecisionError::new(
            dialect,
            cause.status.as_u16(),
            code,
            cause.public_message,
        )),
        None => error(dialect, cause.status, cause.public_message),
    };
    if cause.retry_after {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            HeaderValue::from_static("1"),
        );
    }
    response
}

/// Build a scoring branch through the selected backend's native generate format.
fn bind_native(
    backend: &str,
    model: &str,
    kv_cache_block_size: u32,
    native: &serde_json::Value,
    request_id: &str,
) -> anyhow::Result<Bound> {
    let format = crate::generate_format::named(backend)?;
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

#[cfg(test)]
#[path = "systemone/tests_legacy.rs"]
mod tests;
