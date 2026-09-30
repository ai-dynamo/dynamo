// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

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
    metrics::{CancellationLabels, Endpoint},
    service_v2,
};
use crate::{
    discovery::{ModelManagerError, SystemOneExecutionSelection},
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
}

impl DispatchError {
    fn cancelled() -> Self {
        Self {
            status: StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
            public_message: "request was cancelled",
            detail: "parent request context was cancelled".to_string(),
        }
    }

    fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            public_message: "SGLang scoring worker is unavailable",
            detail: detail.into(),
        }
    }

    fn malformed(detail: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            public_message: "malformed SGLang scoring response",
            detail: detail.into(),
        }
    }
}

pub fn router(state: Arc<service_v2::State>, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    (
        vec![RouteDoc::new(axum::http::Method::POST, &path)],
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
        Err(ModelManagerError::ModelNotFound(_)) => {
            return error(StatusCode::NOT_FOUND, "model is not registered");
        }
        Err(ModelManagerError::ModelUnavailable(_)) => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "model has no eligible aggregate SGLang worker",
            );
        }
        Err(other) => return error(StatusCode::BAD_REQUEST, other.to_string()),
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

    let request_id = super::openai::get_or_create_request_id(&headers);
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
        let (prompt_ids, label_ids) = selection
            .preprocessor
            .render_systemone_question(
                &selection.canonical_model,
                &rendered.content,
                request.chat_template_kwargs.as_ref(),
                &rendered.labels,
            )
            .map_err(|cause| {
                tracing::warn!(question_id = ?question_id, error = %cause, "System One prompt rendering failed");
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    format!(
                        "question {question_id:?}: tokenizer or chat template could not render this request"
                    ),
                )
            })?;
        if context_length != 0 && prompt_ids.len() >= context_length {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                format!(
                    "question {question_id:?}: prompt has {} tokens but model context length is {context_length}",
                    prompt_ids.len()
                ),
            ));
        }
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
) -> Response {
    let mut salt_bytes = [0_u8; 16];
    rand::rng().fill(&mut salt_bytes);
    let cache_salt = URL_SAFE_NO_PAD.encode(salt_bytes);
    let first = branches.remove(0);
    let first_result = match run_branch(
        selection.clone(),
        first,
        request_id.clone(),
        cache_salt.clone(),
        None,
        parent_context.clone(),
    )
    .await
    {
        Ok(result) => result,
        Err(cause) => return dispatch_error(&request_id, &parent_context, cause),
    };
    let placement = first_result.placement;
    let mut results = Vec::with_capacity(branches.len() + 1);
    results.push(first_result);

    let mut siblings = FuturesUnordered::new();
    for branch in branches {
        siblings.push(run_branch(
            selection.clone(),
            branch,
            request_id.clone(),
            cache_salt.clone(),
            Some(placement),
            parent_context.clone(),
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
    tracing::error!(%request_id, error = %cause.detail, "System One scoring failed");
    error(cause.status, cause.public_message)
}

async fn run_branch(
    selection: SystemOneExecutionSelection,
    mut prepared: PreparedBranch,
    request_id: String,
    cache_salt: String,
    pinned: Option<Placement>,
    parent_context: Arc<dyn AsyncEngineContext>,
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
        .map_err(|cause| DispatchError::unavailable(format!("engine dispatch: {cause:#}")))?;
    let stream_context = stream.context();
    parent_context.link_child(stream_context.clone());
    if parent_context.is_killed() {
        stream_context.kill();
        return Err(DispatchError::cancelled());
    }
    finish_branch(
        StartedBranch {
            prepared,
            tracker,
            stream,
        },
        selection.card.runtime_config.data_parallel_start_rank,
        selection.card.runtime_config.data_parallel_size,
        parent_context,
    )
    .await
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
        let frame = frame.map_err(|cause| DispatchError::malformed(cause.to_string()))?;
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
    let input_tokens = terminal
        .get("meta_info")
        .and_then(|meta| meta.get("prompt_tokens"))
        .and_then(|tokens| tokens.as_u64())
        .and_then(|tokens| usize::try_from(tokens).ok())
        .unwrap_or(prepared.prompt_ids.len());
    Ok(BranchResult {
        index: prepared.index,
        question_id: prepared.question_id,
        answer,
        input_tokens,
        placement,
    })
}

#[cfg(test)]
mod tests {
    use std::future::pending;
    use std::sync::Arc;

    use dynamo_runtime::{
        engine::AsyncEngineContextProvider,
        pipeline::{Context, ResponseStream},
    };
    use futures::stream;
    use serde_json::json;
    use tokio::sync::{Semaphore, oneshot};

    use super::{
        Placement, PreparedBranch, StartedBranch, add_input_tokens, finish_branch,
        pin_to_placement, resolve_dp_rank, run_until_killed, spawn_blocking_with_permit,
    };
    use crate::protocols::{
        Annotated,
        common::{llm_backend::LLMEngineOutput, timing::RequestTracker},
        sglang::generate::SglangGenerateRequest,
        systemone::{SystemOneAnswer, SystemOneQuestion, build_native_score_request},
    };

    #[test]
    fn derives_only_a_proven_single_data_parallel_rank() {
        assert_eq!(resolve_dp_rank(None, 7, 1), Some(7));
        assert_eq!(resolve_dp_rank(None, 7, 2), None);
        assert_eq!(resolve_dp_rank(Some(9), 7, 2), Some(9));
    }

    #[test]
    fn cumulative_input_limit_is_checked_without_overflow() {
        assert_eq!(add_input_tokens(7, 5, 12), Ok(12));
        assert!(add_input_tokens(7, 6, 12).is_err());
        assert!(add_input_tokens(usize::MAX, 1, usize::MAX).is_err());
    }

    #[test]
    fn placement_pin_sets_both_worker_and_rank() {
        let native: SglangGenerateRequest =
            serde_json::from_value(build_native_score_request(&[1], &[2], "test-salt").unwrap())
                .unwrap();
        let mut preprocessed = super::super::sglang_generate::preprocessed_request(
            native,
            "test-model",
            None,
            "request-1",
        )
        .unwrap();
        pin_to_placement(
            &mut preprocessed,
            Placement {
                worker_id: 41,
                dp_rank: 3,
            },
        );
        let routing = preprocessed.routing.unwrap();
        assert_eq!(routing.backend_instance_id, Some(41));
        assert_eq!(routing.dp_rank, Some(3));
    }

    #[tokio::test]
    async fn parent_kill_interrupts_pending_branch_work() {
        let parent = Context::new(());
        let context = parent.context();
        let task_context = context.clone();
        let task =
            tokio::spawn(
                async move { run_until_killed(task_context.as_ref(), pending::<()>()).await },
            );
        context.kill();
        assert!(task.await.unwrap().is_none());
    }

    #[tokio::test]
    async fn blocking_preflight_owns_admission_until_it_finishes() {
        let semaphore = Arc::new(Semaphore::new(1));
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let (started_tx, started_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let task = tokio::spawn(spawn_blocking_with_permit(permit, move || {
            started_tx.send(()).unwrap();
            release_rx.blocking_recv().unwrap();
            Ok::<_, ()>(())
        }));
        started_rx.await.unwrap();
        assert!(semaphore.clone().try_acquire_owned().is_err());
        release_tx.send(()).unwrap();
        let (_, returned_permit) = task.await.unwrap().unwrap().unwrap();
        drop(returned_permit);
        assert!(semaphore.try_acquire_owned().is_ok());
    }

    fn prepared_noul_branch() -> PreparedBranch {
        PreparedBranch {
            index: 0,
            question_id: "urgent".to_string(),
            question: serde_json::from_value::<SystemOneQuestion>(json!({
                "type": "noul",
                "instructions": "urgent?"
            }))
            .unwrap(),
            prompt_ids: vec![1, 2],
            label_ids: vec![17, 4],
            native: None,
        }
    }

    #[tokio::test]
    async fn finish_branch_parses_terminal_scores_and_tracker_placement() {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_worker(41, Some(3), "decode");
        let stream = ResponseStream::new(
            Box::pin(stream::iter([Annotated::from_data(LLMEngineOutput {
                engine_data: Some(json!({
                    "sglang_response": {
                        "output_ids": [],
                        "meta_info": {
                            "finish_reason": {"type": "length"},
                            "prompt_tokens": 2,
                            "output_token_ids_logprobs": [[
                                [-0.2, 17, null],
                                [-1.3, 4, null]
                            ]]
                        }
                    }
                })),
                ..Default::default()
            })])),
            Context::new(()).context(),
        );
        let parent = Context::new(());

        let result = finish_branch(
            StartedBranch {
                prepared: prepared_noul_branch(),
                tracker,
                stream,
            },
            0,
            4,
            parent.context(),
        )
        .await;
        let Ok(result) = result else {
            panic!("valid terminal branch should succeed")
        };

        assert_eq!(result.placement.worker_id, 41);
        assert_eq!(result.placement.dp_rank, 3);
        assert_eq!(result.input_tokens, 2);
        assert!(matches!(result.answer, SystemOneAnswer::Noul(_)));
    }

    #[tokio::test]
    async fn finish_branch_stops_when_parent_is_cancelled() {
        let tracker = Arc::new(RequestTracker::new());
        tracker.record_worker(41, Some(0), "decode");
        let stream = ResponseStream::new(
            Box::pin(stream::pending::<Annotated<LLMEngineOutput>>()),
            Context::new(()).context(),
        );
        let parent = Context::new(());
        let parent_context = parent.context();
        parent_context.kill();

        let result = finish_branch(
            StartedBranch {
                prepared: prepared_noul_branch(),
                tracker,
                stream,
            },
            0,
            1,
            parent_context,
        )
        .await;
        let Err(error) = result else {
            panic!("cancelled parent should stop branch completion")
        };

        assert_eq!(error.status.as_u16(), 499);
    }
}
